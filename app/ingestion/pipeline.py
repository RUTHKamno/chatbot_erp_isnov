"""Pipeline d'ingestion : validation → déduplication → extraction → contrôle → chunking → embeddings → index.

Les traitements lourds tournent dans UN SEUL thread d'arrière-plan (un document à la fois) pour ne pas
saturer Ollama / le GPU. Statuts : pending → processing → indexed | needs_review | failed.
"""

import hashlib
import logging
import re
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion import security
from app.ingestion.loaders import LOADERS
from app.ingestion.models import GLOBAL_TENANT, IngestContext
from app.ingestion.registry import Registry, can_see, now
from app.ingestion.settings import get_cfg
from app.ingestion.store import TenantVectorStore, roles_key, tags_key

import unicodedata
from datetime import datetime, timezone

import lancedb
import pyarrow as pa

log = logging.getLogger(__name__)


def format_structured_text(
    role: str, ecran: str, module: str, instructions: str
) -> str:
    """Mise en forme sémantique identique à ingest_guide.py"""
    return (
        f"L'utilisateur disposant du rôle [{role}] est autorisé à effectuer des opérations "
        f"sur l'écran [{ecran}] du module [{module}]. "
        f"Pour réaliser ces actions, la procédure à suivre est la suivante : {instructions}"
    )


def ollama_embedding(text: str) -> List[float]:
    cfg = get_cfg()
    r = requests.post(
        f"{cfg.ollama_base_url}/api/embeddings",
        json={"model": cfg.embed_model, "prompt": text, "keep_alive": "30m"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def _safe_filename(name: str) -> str:
    base = re.sub(r"[^\w.\-]", "_", Path(name or "document").name)
    return base[:120] or "document"


def _position(meta: Dict[str, Any]) -> str:
    if meta.get("page"):
        return f"page {meta['page']}"
    if meta.get("slide"):
        return f"diapositive {meta['slide']}"
    if meta.get("row_start"):
        return f"lignes {meta['row_start']}-{meta['row_end']}"
    return meta.get("section_title") or "document"


TABLE_NAME = "erp_enterprises"

_HEADER_RE = re.compile(
    r"^Tenant\s*:\s*(?P<tenant_id>.*?)\n"
    r"Module\s*:\s*(?P<module>.*?)\n"
    r"Rôles autorisés\s*:\s*(?P<roles>.*?)\n"
    r"Écran\s*:\s*(?P<ecran>.*?)\n"
    r"URL\s*:\s*(?P<url>.*?)\n"
    r"---\n(?P<body>.*)$",
    re.DOTALL,
)
_PAGE_RE = re.compile(r"---\s*\[Page\s+(\d+)\]\s*---")
_ROLE_RE = re.compile(r"^[a-z0-9_\-]+$")


def normalize_role(role: str) -> str:
    """'Service contentieux' -> 'service_contentieux', 'employés' -> 'employes'.
    Doit correspondre EXACTEMENT à la valeur du claim `role` du JWT."""
    r = (
        unicodedata.normalize("NFKD", role)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )
    r = re.sub(r"\s+", "_", r)
    if not _ROLE_RE.match(r):
        raise ValueError(f"rôle invalide : {role!r}")
    return r


def normalize_tenant_id(tenant_id: str) -> str:
    return re.sub(
        r"[^a-z0-9_\-]",
        "_",
        unicodedata.normalize("NFKD", tenant_id)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip(),
    )


def roles_to_key(roles: List[str]) -> str:
    """['pdg','vendeur'] -> '|pdg|vendeur|' (filtrable avec LIKE '%|vendeur|%')."""
    return "|" + "|".join(sorted(set(roles))) + "|"


def parse_formatted_chunk(text: str) -> Dict[str, Any]:
    """Parse un texte produit par convert_and_reformat_document."""
    m = _HEADER_RE.match(text.strip() + "\n" if not text.endswith("\n") else text)
    if not m:
        raise ValueError(
            "en-tête invalide (Tenant / Module / Rôles autorisés / Écran / URL / ---)"
        )
    body = m.group("body").strip()
    page = _PAGE_RE.search(body)
    roles = [normalize_role(r) for r in m.group("roles").split(",") if r.strip()]
    if not roles:
        raise ValueError("aucun rôle autorisé")
    return {
        "tenant_id": normalize_tenant_id(m.group("tenant_id")),
        "module": m.group("module").strip(),
        "roles": roles,
        "ecran": m.group("ecran").strip(),
        "url": m.group("url").strip(),
        "page": int(page.group(1)) if page else None,
        "body": _PAGE_RE.sub("", body).strip(),
    }


def _sql_str(v: str) -> str:
    return v.replace("'", "''")


class IngestionService:
    def __init__(
        self,
        registry: Optional[Registry] = None,
        store: Optional[TenantVectorStore] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        self.registry = registry or Registry()
        self.store = store or TenantVectorStore()
        self.embed = embed_fn or ollama_embedding
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")

    # ------------------------------------------------------------------ soumission (synchrone, rapide)
    def submit_upload(
        self, *, tmp_path: Path, filename: str, ctx: IngestContext
    ) -> Dict[str, Any]:
        cfg = get_cfg()
        ctx.validate()
        tmp_path = Path(tmp_path)
        if tmp_path.stat().st_size > cfg.max_upload_mb * 1024 * 1024:
            raise ValueError(f"Fichier trop volumineux (max {cfg.max_upload_mb} Mo).")
        source_type = security.detect_source_type(tmp_path, filename)

        sha = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha.update(chunk)
        content_hash = sha.hexdigest()

        dup = self.registry.find_duplicate(ctx.tenant_id, ctx.owner_id, content_hash)
        if dup:
            tmp_path.unlink(missing_ok=True)
            return {"doc_id": dup["doc_id"], "status": dup["status"], "duplicate": True}

        doc_id = uuid.uuid4().hex
        dest_dir = cfg.data_dir / "uploads" / ctx.tenant_id / doc_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / _safe_filename(filename)
        shutil.move(str(tmp_path), dest)

        self.registry.create(
            doc_id=doc_id,
            tenant_id=ctx.tenant_id,
            owner_id=ctx.owner_id,
            source_name=_safe_filename(filename),
            source_type=source_type,
            content_hash=content_hash,
            version=1,
            status="pending",
            visibility=ctx.visibility,
            allowed_roles=ctx.allowed_roles,
            doc_type=ctx.doc_type,
            module=ctx.module,
            tags=ctx.tags,
            file_path=str(dest),
        )
        self._executor.submit(self._safe_process, doc_id)
        return {"doc_id": doc_id, "status": "pending", "duplicate": False}

    # ------------------------------------------------------------------ traitement (arrière-plan)
    def _safe_process(self, doc_id: str) -> None:
        try:
            self.process(doc_id)
        except Exception as e:  # noqa: BLE001
            log.exception("Ingestion échouée pour %s", doc_id)
            self.registry.update(
                doc_id, status="failed", status_detail=f"{type(e).__name__}: {e}"[:300]
            )

    def process(self, doc_id: str) -> None:
        cfg = get_cfg()
        doc = self.registry.get_internal(doc_id)
        if doc is None:
            return
        self.registry.update(doc_id, status="processing")

        # 1. Extraction
        extracted = LOADERS[doc["source_type"]](Path(doc["file_path"]))

        # 2. Contrôle de sécurité : le contenu est une donnée NON FIABLE destinée à un prompt
        full_text = "\n".join(b.text for b in extracted.blocks)
        hits = security.scan_injection(full_text)
        if hits:
            self.registry.update(
                doc_id,
                status="needs_review",
                warnings=extracted.warnings,
                stats=extracted.stats,
                status_detail=f"Contenu suspect (tentative d'injection de prompt ?) : {hits[0]!r}",
            )
            return

        # 3. Contrôle qualité
        if not extracted.blocks:
            self.registry.update(
                doc_id,
                status="needs_review",
                warnings=extracted.warnings,
                stats=extracted.stats,
                status_detail="Aucun texte extractible (document scanné ? l'OCR arrive en phase 2).",
            )
            return

        # 4. Chunking (les blocs sont déjà sémantiques : page / diapositive / groupe de lignes)
        # 3b. MÉTADONNÉES LUES DANS LE DOCUMENT : validation stricte (fail-closed)
        uploader_tenant = doc[
            "tenant_id"
        ]  # entreprise de l'administrateur, issue du JWT à l'envoi

        def refuse(status: str, message: str) -> None:
            log.warning("Ingestion %s : %s - %s", doc_id, status, message)
            self.registry.update(
                doc_id,
                status=status,
                warnings=extracted.warnings,
                stats=extracted.stats,
                status_detail=message[:500],
            )

        declared = {
            str(b.meta["tenant_id"]).strip().lower()
            for b in extracted.blocks
            if b.meta.get("tenant_id")
        }
        if len(declared) > 1:
            return refuse(
                "rejected",
                f"Le document déclare plusieurs entreprises : {sorted(declared)}.",
            )
        tenant = declared.pop() if declared else uploader_tenant
        if tenant != uploader_tenant:
            # Un administrateur ne peut écrire QUE dans l'index de sa propre entreprise (jamais 'global' non plus)
            return refuse(
                "rejected",
                f"Le document déclare l'entreprise « {tenant} » mais vous administrez "
                f"« {uploader_tenant} » : ingestion refusée.",
            )
        if not any(b.meta.get("tenant_id") for b in extracted.blocks):
            extracted.warnings.append(
                "Entreprise non précisée dans le document : celle de l'administrateur est utilisée."
            )

        problems = []
        for b in extracted.blocks:
            label = (
                b.meta.get("ecran") or b.meta.get("section_title") or _position(b.meta)
            )
            roles = [
                str(r).strip().lower()
                for r in (b.meta.get("roles") or [])
                if str(r).strip()
            ]
            if not roles:
                problems.append(
                    f"« {label} » : aucun rôle autorisé (ajoutez « Rôles autorisés : ... »)"
                )
                continue
            try:
                b.meta["roles"] = sorted({security.check_role(r) for r in roles})
            except ValueError:
                problems.append(
                    f"« {label} » : rôle invalide dans {roles} (identifiants du JWT : minuscules, chiffres, - et _)"
                )
        if problems:
            more = f" (+{len(problems) - 5} autre(s))" if len(problems) > 5 else ""
            return refuse("needs_review", " ; ".join(problems[:5]) + more)

        # 4. Chunking + lignes d'index : UNE ligne par chunk, les rôles autorisés sont portés par la ligne
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        base_quality = "partial" if extracted.partial else "ok"
        ingested_at = now()
        prev = self.registry.find_previous_version(
            tenant, doc["owner_id"], doc["source_name"], doc_id
        )
        version = prev["version"] + 1 if prev else 1

        rows: List[Dict[str, Any]] = []
        idx = 0
        for block in extracted.blocks:
            m = block.meta
            roles = m["roles"]
            visibility = "tenant" if "all" in roles else "roles"
            for raw_chunk in splitter.split_text(block.text):
                text = self._chunk_text(doc["source_name"], m, roles, raw_chunk)
                rows.append(
                    {
                        "id": f"{doc_id}_c{idx}",
                        "text": text,
                        "vector": self._embed_retry(text),
                        "tenant_id": tenant,
                        "visibility": visibility,
                        "allowed_roles": roles_key(
                            [] if visibility == "tenant" else roles
                        ),
                        "owner_id": doc["owner_id"],
                        "doc_id": doc_id,
                        "doc_version": version,
                        "content_hash": doc["content_hash"],
                        "source_type": doc["source_type"],
                        "source_name": doc["source_name"],
                        "source_url": None,
                        "ingested_at": ingested_at,
                        "ingested_by": doc["owner_id"],
                        "page": m.get("page"),
                        "slide": m.get("slide"),
                        "row_start": m.get("row_start"),
                        "row_end": m.get("row_end"),
                        "section_title": m.get("section_title"),
                        "chunk_index": idx,
                        "extraction_method": extracted.extraction_method,
                        "confidence": extracted.confidence,
                        "quality_flag": base_quality,
                        "doc_type": m.get("doc_type") or "document",
                        "module": m.get("module") or "",
                        "ecran": m.get("ecran") or "",
                        "url": m.get("url") or "",
                        "tags": "",
                    }
                )
                idx += 1

        if not rows:
            return refuse(
                "needs_review", "Aucun contenu à indexer après lecture des métadonnées."
            )

        # 5. Indexation (tout ou rien) dans la table de l'entreprise, puis remplacement de l'ancienne version
        self.store.add_rows(tenant, rows)
        if prev:
            self.store.delete_document(tenant, prev["doc_id"])
            self.registry.update(prev["doc_id"], status="superseded")
        self.registry.update(
            doc_id,
            status="indexed",
            version=version,
            n_chunks=len(rows),
            warnings=extracted.warnings,
            stats=extracted.stats,
            status_detail=(
                "Indexé partiellement : voir les avertissements."
                if extracted.partial
                else ""
            ),
        )

    @staticmethod
    def _chunk_text(
        source_name: str, meta: Dict[str, Any], roles: List[str], chunk: str
    ) -> str:
        """Même mise en forme que ingest_guide.py quand l'écran / le module sont connus ; sinon en-tête de source."""
        ecran, module = meta.get("ecran") or "", meta.get("module") or ""
        if ecran or module:
            label = "tous les rôles" if "all" in roles else ", ".join(roles)
            return format_structured_text(
                role=label, ecran=ecran or "-", module=module or "-", instructions=chunk
            )
        return f"[Source : {source_name} — {_position(meta)}]\n{chunk}"

    def _embed_retry(self, text: str) -> List[float]:
        last: Optional[Exception] = None
        for attempt in range(get_cfg().embed_retries + 1):
            try:
                vec = self.embed(text)
                if vec:
                    return vec
            except Exception as e:  # noqa: BLE001
                last = e
            time.sleep(min(2 * attempt, 5))
        raise RuntimeError(f"Embedding indisponible : {last}")

        # ------------------------------------------------------------------ insertion de textes déjà reformatés

    def _open_table(self, dim: int):
        cfg = get_cfg()
        db = lancedb.connect(str(getattr(cfg, "lancedb_uri", cfg.data_dir / "lancedb")))
        if TABLE_NAME in db.table_names():
            return db.open_table(TABLE_NAME)
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("tenant_id", pa.string()),
                pa.field("allowed_roles", pa.string()),  # '|pdg|vendeur|'
                pa.field(
                    "visibility", pa.string()
                ),  # 'tenant' (tous les rôles) | 'roles'
                pa.field("module", pa.string()),
                pa.field("ecran", pa.string()),
                pa.field("url", pa.string()),
                pa.field("source_name", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("page", pa.int32()),
                pa.field("chunk_index", pa.int32()),
                pa.field("ingested_at", pa.string()),
                pa.field("doc_type", pa.string()),  # <--- CHAMP À AJOUTER ICI
            ]
        )
        return db.create_table(TABLE_NAME, schema=schema)

    # Fichier : app/ingestion/pipeline.py

    def ingest_formatted_texts(
        self, *, texts: List[str], filename: str, tenant_id: str
    ) -> Dict[str, Any]:
        """
        Insère dans LanceDB (table `erp_enterprises`) :
        1. Les blocs RAG vectorisés.
        2. Les entrées RBAC brutes (doc_type = 'rbac_role', vector = [0.0...]) pour l'Évaluateur.
        """
        cfg = get_cfg()
        tenant_id = normalize_tenant_id(tenant_id)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        doc_id = uuid.uuid4().hex
        ingested_at = now()
        source_name = _safe_filename(filename)

        # 1. Parsing + validation de tous les blocs
        parsed, problems = [], []
        for i, t in enumerate(texts, 1):
            try:
                p = parse_formatted_chunk(t)
                if p["tenant_id"] != tenant_id:
                    raise ValueError(
                        f"tenant « {p['tenant_id']} » différent de « {tenant_id} »"
                    )
                parsed.append(p)
            except ValueError as e:
                problems.append(f"chunk {i} : {e}")

        if problems:
            return {"status": "needs_review", "n_chunks": 0, "problems": problems[:10]}

        # 2. Construction des lignes RAG vectorisées
        rows: List[Dict[str, Any]] = []
        idx = 0

        # Structure pour agréger la matrice RBAC (Rôle -> Liste des écrans)
        rbac_matrix: Dict[str, set] = {}

        for p in parsed:
            visibility = "tenant" if "all" in p["roles"] else "roles"

            # Consolidation RBAC pour l'évaluateur
            screen_or_module = p["ecran"] or p["module"]
            if screen_or_module:
                for role in p["roles"]:
                    role_clean = role.lower().strip()
                    if role_clean not in rbac_matrix:
                        rbac_matrix[role_clean] = set()
                    rbac_matrix[role_clean].add(screen_or_module)

            # Ingestion RAG
            for piece in splitter.split_text(p["body"]):
                text = format_structured_text(
                    role=(
                        "tous les rôles"
                        if visibility == "tenant"
                        else ", ".join(p["roles"])
                    ),
                    ecran=p["ecran"] or "-",
                    module=p["module"] or "-",
                    instructions=piece,
                )
                rows.append(
                    {
                        "id": f"{doc_id}_c{idx}",
                        "text": text,
                        "vector": self._embed_retry(text),
                        "tenant_id": tenant_id,
                        "allowed_roles": roles_to_key(p["roles"]),
                        "visibility": visibility,
                        "module": p["module"],
                        "ecran": p["ecran"],
                        "url": p["url"],
                        "source_name": source_name,
                        "doc_id": doc_id,
                        "page": p["page"],
                        "chunk_index": idx,
                        "ingested_at": ingested_at,
                        "doc_type": "document",  # Identifie les Chunks RAG
                    }
                )
                idx += 1

        if not rows:
            return {
                "status": "needs_review",
                "n_chunks": 0,
                "problems": ["aucun contenu à indexer"],
            }

        dim = len(rows[0]["vector"])

        # 3. Construction des lignes RBAC brutes (Zero-Vector) pour l'Évaluateur
        rbac_rows: List[Dict[str, Any]] = []
        zero_vector = [0.0] * dim

        for role, screens in rbac_matrix.items():
            screens_str = ", ".join(sorted(list(screens)))
            rbac_rows.append(
                {
                    "id": f"rbac_{tenant_id}_{role}",
                    "text": screens_str,  # Liste des écrans autorisés
                    "vector": zero_vector,  # Pas d'embedding (0 cost Ollama)
                    "tenant_id": tenant_id,
                    "allowed_roles": f"|{role}|",
                    "visibility": "tenant",
                    "module": "RBAC_SYSTEM",
                    "ecran": "RBAC_SYSTEM",
                    "url": "",
                    "source_name": "rbac_matrix",
                    "doc_id": f"rbac_{tenant_id}",
                    "page": 0,
                    "chunk_index": 0,
                    "ingested_at": ingested_at,
                    "doc_type": "rbac_role",  # Lu par evaluator.py
                }
            )

        # 4. Écriture dans LanceDB (table `erp_enterprises`)
        table = self._open_table(dim)

        # Supprimer les anciennes entrées pour le même fichier et le même tenant
        table.delete(
            f"tenant_id = '{_sql_str(tenant_id)}' AND source_name = '{_sql_str(source_name)}'"
        )

        # Supprimer l'ancienne matrice RBAC de ce tenant si on écrit la nouvelle
        if rbac_rows:
            table.delete(
                f"tenant_id = '{_sql_str(tenant_id)}' AND doc_type = 'rbac_role'"
            )

        # Insertion atomique
        all_rows_to_insert = rows + rbac_rows
        table.add(all_rows_to_insert)

        return {
            "status": "indexed",
            "doc_id": doc_id,
            "table": TABLE_NAME,
            "tenant_id": tenant_id,
            "rag_chunks": len(rows),
            "rbac_roles_indexed": len(rbac_rows),
        }

    # ------------------------------------------------------------------ recherche filtrée par rôle
    def search(
        self, *, tenant_id: str, role: str, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        cfg = get_cfg()
        db = lancedb.connect(str(getattr(cfg, "lancedb_uri", cfg.data_dir / "lancedb")))
        table = db.open_table(TABLE_NAME)
        role = normalize_role(role)
        flt = (
            f"tenant_id = '{_sql_str(normalize_tenant_id(tenant_id))}' AND "
            f"(allowed_roles LIKE '%|all|%' OR allowed_roles LIKE '%|{role}|%')"
        )
        return (
            table.search(self._embed_retry(query))
            .where(flt, prefilter=True)
            .limit(k)
            .to_list()
        )

    # ------------------------------------------------------------------ gestion
    def list_documents(
        self, *, tenant_id: str, user_id: str, role: str
    ) -> List[Dict[str, Any]]:
        cfg = get_cfg()
        return [
            d
            for d in self.registry.list_tenant(tenant_id)
            if can_see(d, user_id, role, cfg.admin_roles)
        ]

    def get_document(
        self, *, tenant_id: str, doc_id: str, user_id: str, role: str
    ) -> Optional[Dict[str, Any]]:
        d = self.registry.get(tenant_id, doc_id)
        if d is None or d["status"] in ("deleted", "superseded"):
            return None
        if not can_see(d, user_id, role, get_cfg().admin_roles):
            return None  # même réponse qu'un document inexistant : pas de fuite d'existence
        return d

    def delete_document(
        self, *, tenant_id: str, doc_id: str, user_id: str, role: str
    ) -> bool:
        d = self.get_document(
            tenant_id=tenant_id, doc_id=doc_id, user_id=user_id, role=role
        )
        if d is None:
            return False
        if d["owner_id"] != user_id and role not in get_cfg().admin_roles:
            raise PermissionError(
                "Seul le propriétaire ou un administrateur peut supprimer ce document."
            )
        self.store.delete_document(tenant_id, doc_id)
        self.registry.update(doc_id, status="deleted")
        if d.get("file_path"):
            shutil.rmtree(Path(d["file_path"]).parent, ignore_errors=True)
        return True


# Dans app/ingestion/pipeline.py


def save_tenant_rbac_to_lancedb(
    service: IngestionService, tenant_id: str, extracted_metadata: list[dict]
):
    """
    Insère/met à jour les règles RBAC directement dans LanceDB sous forme de données brutes (sans embedding).
    """
    # 1. Regrouper les écrans par rôle
    rbac_matrix: dict[str, set[str]] = {}
    for meta in extracted_metadata:
        roles = meta.get("roles", ["all"])
        ecran = meta.get("ecran") or meta.get("screen")
        module = meta.get("module")
        screen = ecran or module
        if not screen:
            continue
        for role in roles:
            r_norm = role.lower().strip()
            if r_norm not in rbac_matrix:
                rbac_matrix[r_norm] = set()
            rbac_matrix[r_norm].add(screen)

    if not rbac_matrix:
        return

    # 2. Préparer les lignes pour LanceDB
    # On utilise un vecteur nul de dimension 768 (ou la dimension configurée)
    dim = get_cfg().embed_dim if hasattr(get_cfg(), "embed_dim") else 768
    zero_vector = [0.0] * dim
    ingested_at = now()

    rows = []
    for role, screens in rbac_matrix.items():
        screens_str = ", ".join(sorted(list(screens)))
        rows.append(
            {
                "id": f"rbac_{tenant_id}_{role}",
                "text": screens_str,  # Liste brute des écrans autorisés
                "vector": zero_vector,  # Pas de vectorisation nécessaire !
                "tenant_id": tenant_id,
                "allowed_roles": f"|{role}|",
                "visibility": "tenant",
                "module": "RBAC_SYSTEM",
                "ecran": "RBAC_SYSTEM",
                "url": "",
                "source_name": "rbac_matrix",
                "doc_id": f"rbac_{tenant_id}",
                "page": 0,
                "chunk_index": 0,
                "ingested_at": ingested_at,
                "doc_type": "rbac_role",  # Tag spécifique pour l'évaluateur
            }
        )

    # 3. Insertion / Remplacement dans LanceDB
    table = service._open_table(dim)
    # Nettoyer les anciennes règles RBAC de ce tenant
    table.delete(f"tenant_id = '{_sql_str(tenant_id)}' AND doc_type = 'rbac_role'")
    table.add(rows)
    log.info(
        f" Droits RBAC enregistrés dans LanceDB pour le tenant '{tenant_id}' ({len(rows)} rôles)."
    )


_service: Optional[IngestionService] = None


def get_default_service() -> IngestionService:
    global _service
    if _service is None:
        _service = IngestionService()
    return _service
