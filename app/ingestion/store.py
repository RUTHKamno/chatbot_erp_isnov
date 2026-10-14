"""Base vectorielle multi-entreprises (LanceDB) : UNE TABLE PAR ENTREPRISE + une table 'global'.

Isolation :
  1. physique  : table docs_<tenant> (une erreur de filtre ne peut pas mélanger deux clients)
  2. logique   : tenant_id + visibilité (tenant / roles / private) dans chaque requête
  3. contrôle  : après la recherche, tout résultat hors tenant déclenche IsolationError
"""

import logging
import lancedb
import pyarrow as pa
import unicodedata


from app.config import get_settings
from app.ingestion import security
from app.ingestion.models import GLOBAL_TENANT
from app.ingestion.security import IsolationError
from app.ingestion.settings import get_cfg
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

TABLE_NAME = "erp_enterprises"
GUIDE_TABLE_NAME = get_settings().lancedb_table


def normalize_string(text: str) -> str:
    """Transforme 'Écran / Ouvriér' en 'ecran / ouvrier'."""
    if not text:
        return ""
    nfkd_form = unicodedata.normalize("NFKD", text)
    only_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return only_ascii.lower().strip()


def table_name(tenant_id: str) -> str:
    return f"docs_{tenant_id}"


def roles_key(roles: List[str]) -> str:
    """['financier','direction'] -> '|direction|financier|' (filtrable avec LIKE '%|role|%')."""
    return "|" + "|".join(sorted(roles)) + "|" if roles else ""


def tags_key(tags: List[str]) -> str:
    return "|" + "|".join(tags) + "|" if tags else ""


def _schema(dim: int) -> pa.Schema:
    s, i, f = pa.string(), pa.int32(), pa.float32()
    return pa.schema(
        [
            pa.field("id", s),
            pa.field("text", s),
            pa.field("vector", pa.list_(f, dim)),
            # isolation et droits
            pa.field("tenant_id", s),
            pa.field("visibility", s),
            pa.field("allowed_roles", s),
            pa.field("owner_id", s),
            # provenance
            pa.field("doc_id", s),
            pa.field("doc_version", i),
            pa.field("content_hash", s),
            pa.field("source_type", s),
            pa.field("source_name", s),
            pa.field("source_url", s),
            pa.field("ingested_at", s),
            pa.field("ingested_by", s),
            # position
            pa.field("page", i),
            pa.field("slide", i),
            pa.field("row_start", i),
            pa.field("row_end", i),
            pa.field("start_s", f),
            pa.field("end_s", f),
            pa.field("section_title", s),
            pa.field("chunk_index", i),
            # qualité
            pa.field("extraction_method", s),
            pa.field("confidence", f),
            pa.field("quality_flag", s),
            pa.field("language", s),
            # métier
            pa.field("doc_type", s),
            pa.field("module", s),
            pa.field("ecran", s),
            pa.field("tags", s),
            pa.field("url", s),
        ]
    )


class TenantVectorStore:
    def __init__(self, uri: Optional[str] = None):
        self.db = lancedb.connect(uri or get_cfg().lancedb_uri)

    def init_table(self, data: List[Dict[str, Any]]):
        """Initialise ou remplace la table historique des guides ERP."""
        return self.db.create_table(GUIDE_TABLE_NAME, data=data, mode="overwrite")

    # ------------------------------------------------------------ utilitaires
    def _open(self, name: str):
        try:
            return self.db.open_table(name)
        except (ValueError, FileNotFoundError):
            return None

    # ------------------------------------------------------------ écriture
    def add_rows(self, tenant_id: str, rows: List[Dict[str, Any]]) -> int:
        tenant_id = security.check_tenant(tenant_id, allow_global=True)
        if not rows:
            return 0
        if any(r.get("tenant_id") != tenant_id for r in rows):
            raise IsolationError(
                "Une ligne à indexer porte un tenant_id différent de la table cible."
            )
        schema = _schema(len(rows[0]["vector"]))
        data = pa.Table.from_pylist(rows, schema=schema)
        tbl = self._open(table_name(tenant_id))
        if tbl is None:
            tbl = self.db.create_table(table_name(tenant_id), schema=schema)
        tbl.add(data)
        return len(rows)

    def delete_document(self, tenant_id: str, doc_id: str) -> None:
        tenant_id = security.check_tenant(tenant_id, allow_global=True)
        doc_id = security.check_doc_id(doc_id)
        tbl = self._open(table_name(tenant_id))
        if tbl is not None:
            tbl.delete(f"doc_id = '{doc_id}' AND tenant_id = '{tenant_id}'")

    def drop_tenant(self, tenant_id: str) -> None:
        """Droit à l'effacement : supprime TOUTES les données d'une entreprise."""
        tenant_id = security.check_tenant(tenant_id)
        try:
            self.db.drop_table(table_name(tenant_id))
        except (ValueError, FileNotFoundError):
            pass

    # ------------------------------------------------------------ recherche
    def search_guide(
        self,
        query_vector: List[float],
        user_role: str,
        tenant_id: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Recherche dans les guides ERP, avec filtrage par rôle et tenant si présent."""
        table = self._open(GUIDE_TABLE_NAME)
        if table is None or not query_vector:
            return []

        user_role = security.check_role(user_role)
        filters = [f"(role_autorise = '{user_role}' OR role_autorise = 'all')"]
        columns = set(table.schema.names)
        if tenant_id and "tenant_id" in columns:
            tenant_id = security.check_tenant(tenant_id)
            tenant_filter = f"(tenant_id = '{tenant_id}'"
            if "is_global" in columns:
                tenant_filter += " OR is_global = True"
            tenant_filter += ")"
            filters.append(tenant_filter)
        elif "is_global" in columns and "tenant_id" in columns:
            filters.append("(is_global = True OR tenant_id IS NULL)")

        results = (
            table.search(query_vector)
            .where(" AND ".join(filters), prefilter=True)
            .limit(top_k)
            .to_list()
        )
        for row in results:
            row.pop("vector", None)
        return results

    @staticmethod
    def _where(tid: str, user_id: str, role: str, source_types, doc_types) -> str:
        visible = (
            "visibility = 'tenant' "
            f"OR (visibility = 'roles' AND allowed_roles LIKE '%|{role}|%') "
            f"OR (visibility = 'private' AND owner_id = '{user_id}')"
        )
        where = f"tenant_id = '{tid}' AND ({visible})"
        if source_types:
            vals = ", ".join(
                f"'{security.check_slug(v, 'source_type')}'" for v in source_types
            )
            where += f" AND source_type IN ({vals})"
        if doc_types:
            vals = ", ".join(
                f"'{security.check_slug(v, 'doc_type')}'" for v in doc_types
            )
            where += f" AND doc_type IN ({vals})"
        return where

    def search_enterprises(
        self, query_vector: List[float], tenant_id: str, user_role: str, top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Recherche dans les blocs d'entreprise, filtrée par tenant et rôles autorisés.
        """
        tenant_id = security.check_tenant(tenant_id)
        user_role = security.check_role(user_role)

        print(
            f"[VectorStore] Recherche pour tenant_id={tenant_id}, user_role={user_role}, top_k={top_k}"
        )

        tbl = self._open(TABLE_NAME)
        if tbl is None or not query_vector:
            return []

        escaped_tenant = tenant_id.replace("'", "''")
        print(
            f"[VectorStore] Filtre SQL : tenant_id='{escaped_tenant}', user_role='{user_role}'"
        )
        norm_role = normalize_string(user_role)

        if norm_role in ("all", "*", ""):
            where_clause = f"tenant_id = '{escaped_tenant}'"
        else:

            where_clause = (
                f"tenant_id = '{escaped_tenant}' AND "
                f"(allowed_roles LIKE '%|{user_role}|%' OR allowed_roles LIKE '%|all|%')"
            )
        print(f"[VectorStore] WHERE clause: {where_clause}")
        results = (
            tbl.search(query_vector)
            .metric("cosine")
            .where(where_clause, prefilter=True)
            .limit(top_k)
            .to_list()
        )
        for row in results:
            row.pop("vector", None)
        return results

    def search(
        self,
        *,
        tenant_id: str,
        user_id: str,
        user_role: str,
        query_vector: List[float],
        top_k: int = 20,
        include_global: bool = True,
        source_types: Optional[List[str]] = None,
        doc_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        # Tous les paramètres sont obligatoires et validés : pas de recherche "sans tenant".
        tenant_id = security.check_tenant(tenant_id)
        user_id = security.check_user(user_id)
        user_role = security.check_role(user_role)
        if not query_vector:
            return []

        rows: List[Dict[str, Any]] = []
        for tid in [tenant_id] + ([GLOBAL_TENANT] if include_global else []):
            tbl = self._open(table_name(tid))
            if tbl is None:
                continue
            where = self._where(tid, user_id, user_role, source_types, doc_types)
            res = (
                tbl.search(query_vector)
                .metric("cosine")
                .where(where, prefilter=True)
                .limit(top_k)
                .to_list()
            )
            rows.extend(res)

        for r in rows:  # contrôle de sortie (défense en profondeur)
            if r.get("tenant_id") not in (tenant_id, GLOBAL_TENANT):
                log.critical(
                    "FUITE INTER-ENTREPRISES bloquée : demandé=%s reçu=%s",
                    tenant_id,
                    r.get("tenant_id"),
                )
                raise IsolationError(
                    "Résultat d'une autre entreprise détecté : requête bloquée."
                )
            r.pop("vector", None)
        rows.sort(key=lambda r: r.get("_distance", 1.0))
        return rows[:top_k]


_store: Optional[TenantVectorStore] = None


def get_store() -> TenantVectorStore:
    global _store
    if _store is None:
        _store = TenantVectorStore()
    return _store


vector_store = get_store()
