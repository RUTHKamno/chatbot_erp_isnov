"""API d'ingestion. Les identifiants d'accès viennent du JWT; l'upload accepte son tenant en formulaire."""

import json
import logging
import re
import tempfile
import traceback
import uuid  # noqa: UP035
from dataclasses import dataclass
from pathlib import Path
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app import ingestion
from app.core import db_manager
from app.core.cube_semantic_layer import push_cube_dev_context, test_cube_connection
from app.ingestion import security
from app.ingestion.converter import convert_and_reformat_document
from app.ingestion.loaders import LOADERS
from app.ingestion.models import IngestContext
from app.ingestion.pipeline import SemanticFileTooLargeError, normalize_tenant_id
from app.ingestion.security import UnsupportedFile
from app.ingestion.settings import get_cfg
from app.ingestion.web_scrapy import (
    ScrapingBlockedException,
    ScrapingCancelled,
    crawl_and_accumulate_sync,
)

# Logger dédié pour les tests et erreurs de conversion
logger = logging.getLogger("ingestion.test_jalon1")
logger.setLevel(logging.INFO)


# >>> À ADAPTER : remplace cette fonction par ton décodage JWT existant (elle doit renvoyer les claims) <<<
def current_claims(authorization: str | None = Header(None)) -> dict:
    import jwt  # PyJWT

    from app.config import get_settings

    if not authorization:
        raise HTTPException(401, "Authentification requise.")
    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token.strip():
        raise HTTPException(401, "Authentification Bearer requise.")

    s = get_settings()
    try:
        return jwt.decode(token.strip(), s.jwt_secret_key, algorithms=[s.jwt_algorithm])
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Token invalide ou expiré")


@dataclass
class Identity:
    tenant_id: str
    user_id: str
    role: str


class StartDiscoveryRequest(BaseModel):
    start_url: str = Field(..., description="URL initiale à explorer")
    tenant_id: str = Field(..., description="Identifiant du tenant")
    session_id: str | None = Field(
        None, description="Identifiant fourni par le client pour pouvoir annuler."
    )


class ContinueDiscoveryRequest(BaseModel):
    session_id: str = Field(
        ..., description="Identifiant de la session d'exploration en cours"
    )
    tenant_id: str = Field(..., description="Identifiant du tenant")
    continue_exploration: bool = Field(
        ...,
        description="True pour scrapper les URLs sélectionnées et continuer ; False pour stopper et tout ingérer dans LanceDB.",
    )
    selected_urls: List[str] = Field(
        default_factory=list,
        description="Liste des URLs cochées par l'utilisateur pour le prochain niveau.",
    )
    ingest: bool = Field(
        True, description="Si continue_exploration=False, effectue l'ingestion LanceDB"
    )


def identity(claims: dict = Depends(current_claims)) -> Identity:
    tenant, sub, role = claims.get("tenant_id"), claims.get("sub"), claims.get("role")
    if not (tenant and sub and role):
        raise HTTPException(
            403, "Claims JWT incomplets : tenant_id, sub et role sont obligatoires."
        )
    if role.lower() not in {r.lower() for r in get_cfg().admin_roles}:
        raise HTTPException(403, "Droit d'ingestion requis (rôle administrateur).")
    return Identity(tenant, sub, role)


def _ensure_tenant(identity: Identity, tenant_id: str) -> None:
    if identity.tenant_id.casefold() != tenant_id.casefold():
        raise HTTPException(403, "Accès interdit à l’entreprise demandée.")


router = APIRouter(
    prefix="/ingest",
    tags=["ingestion"],
    dependencies=[Depends(identity)],
)


def get_service():
    from app.ingestion.pipeline import get_default_service

    return get_default_service()


def _validate_discovery_session_id(session_id: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise HTTPException(400, "Identifiant de session d'exploration invalide.")


def _cancel_file(session_dir: Path) -> Path:
    return session_dir / "cancel.requested"


def _next_chunk_or_none(chunks):
    return next(chunks, None)


def _split(csv_value: str) -> List[str]:
    return [v.strip() for v in (csv_value or "").split(",") if v.strip()]


def _public(d: dict) -> dict:
    keys = (
        "doc_id",
        "source_name",
        "source_type",
        "version",
        "status",
        "status_detail",
        "n_chunks",
        "visibility",
        "allowed_roles",
        "doc_type",
        "module",
        "tags",
        "warnings",
        "created_at",
        "updated_at",
    )
    return {k: d[k] for k in keys}


# Dans app/api/ingestion.py

"""
@router.post("/upload", status_code=202, summary="Ingérer un document ERP")
async def upload(
    file: UploadFile = File(...),
    tenant_id: str = Form(..., description="Identifiant du tenant"),
    format_text: bool = Form(True, description="True pour reformatage IA, False pour textsplit direct"),
    ident: Identity = Depends(identity),
    service=Depends(get_service),
):
    _ensure_tenant(ident, tenant_id)
    cfg = get_cfg()
    tmp_dir = cfg.data_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"upload_{file.filename}"

    try:
        with open(tmp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # 1. Convertir et formater les blocs avec l'IA / Converter
        formatted_blocks = list(
            convert_and_reformat_document(
                file_path=tmp_path,
                filename=file.filename or "document",
                tenant_id=tenant_id,
                pages_per_chunk=5,
            )
        )

        if not formatted_blocks:
            raise HTTPException(
                400, "Aucun contenu extractible trouvé dans le document."
            )

        # 2. Ingestion synchrone/arrière-plan dans LanceDB (Chunks RAG + Matrice RBAC)
        result = await run_in_threadpool(
            service.ingest_formatted_texts,
            texts=formatted_blocks,
            filename=file.filename or "document",
            tenant_id=tenant_id,
        )

        return {
            "status": "success",
            "filename": file.filename,
            "tenant_id": tenant_id,
            "ingestion": result,
            "block_inserted": formatted_blocks,
        }

    except UnsupportedFile as e:
        raise HTTPException(415, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

"""


@router.post("/upload", status_code=202, summary="Ingérer un document ERP")
async def upload(
    file: UploadFile = File(...),
    tenant_id: str = Form(..., description="Identifiant du tenant"),
    format_text: bool = Form(
        True, description="True pour reformatage IA, False pour textsplit direct"
    ),
    ident: Identity = Depends(identity),
    service=Depends(get_service),
):
    _ensure_tenant(ident, tenant_id)
    cfg = get_cfg()
    tmp_dir = cfg.data_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"upload_{file.filename}"

    try:
        with open(tmp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        if format_text:
            # 1. Comportement standard : Convertir et formater avec l'IA
            formatted_blocks = list(
                convert_and_reformat_document(
                    file_path=tmp_path,
                    filename=file.filename or "document",
                    tenant_id=tenant_id,
                    pages_per_chunk=5,
                )
            )
            if not formatted_blocks:
                raise HTTPException(
                    400, "Aucun contenu extractible trouvé dans le document."
                )

            result = await run_in_threadpool(
                service.ingest_formatted_texts,
                texts=formatted_blocks,
                filename=file.filename or "document",
                tenant_id=tenant_id,
            )
        else:
            # 2. Comportement direct glossaire/description : Stockage direct dans tenant_semantic_models
            source_type = security.detect_source_type(
                tmp_path, file.filename or "document"
            )
            if source_type not in LOADERS:
                raise HTTPException(
                    415, f"Type de fichier non supporté : {source_type}"
                )

            extracted = LOADERS[source_type](tmp_path)
            if not extracted.blocks:
                raise HTTPException(
                    400, "Aucun contenu extractible trouvé dans le document."
                )

            full_text = "\n\n".join(b.text for b in extracted.blocks if b.text.strip())
            if not full_text.strip():
                raise HTTPException(400, "Le texte extrait est vide.")

            # On réutilise le fichier temporaire ou un objet fichier en mémoire pour indexer
            from io import BytesIO

            file_obj_mem = BytesIO(full_text.encode("utf-8"))

            result = await run_in_threadpool(
                service.index_glossary_file,
                file_obj=file_obj_mem,
                filename=file.filename or "dictionnaire.md",
                tenant_id=tenant_id,
            )

        return {
            "status": "success",
            "filename": file.filename,
            "tenant_id": tenant_id,
            "format_text": format_text,
            "ingestion": result,
        }

    except UnsupportedFile as e:
        raise HTTPException(415, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


class TextIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    text: str = Field(
        min_length=1, max_length=200_000
    )  # le texte contient ses propres lignes « Tenant : », « Rôles autorisés : »...


# -- base de données
async def _db_call(fn, *args, **kwargs):
    try:
        logger.debug("Appel de la fonction DB %s", fn.__name__)
        return await run_in_threadpool(fn, *args, **kwargs)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        logger.exception("Erreur base de données")
        raise HTTPException(
            502, "Erreur d'accès à la base de données (voir les logs serveur)."
        )


def _check_db_type(db_type: str) -> None:
    if db_type != "postgres":
        raise HTTPException(400, "Seul PostgreSQL est géré pour l'instant.")
    return db_type


@router.post("/db/test", summary="[BD 1] Tester une connexion")
async def db_test(
    db_type: str = Form("postgres"),
    host: str = Form(...),
    port: int = Form(5432),
    database: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    # ident: Identity = Depends(identity),
):
    db_type = _check_db_type(db_type)
    await _db_call(
        db_manager.test_connection,
        db_type=db_type,
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
    )
    return {"status": "ok", "message": "Connexion réussie."}


@router.post(
    "/db/connections", status_code=201, summary="[BD 2] Enregistrer une connexion"
)
async def db_save(
    tenant_id: str = Form(..., description="Identifiant du tenant"),
    name: str = Form(..., description="Nom de la connexion"),
    db_type: str = Form("postgres"),
    host: str = Form(...),
    port: int = Form(5432),
    database: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    show_all_schemas: bool = Form(False, description="Afficher toutes les tables"),
    ident: Identity = Depends(identity),
):

    _ensure_tenant(ident, tenant_id)
    tenant_id = normalize_tenant_id(tenant_id)
    connection_id = await _db_call(
        db_manager.save_connection,
        tenant_id=tenant_id,
        name=name.strip(),
        db_type=db_type,
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        show_all_schemas=show_all_schemas,
    )
    return {"connection_id": connection_id, "name": name.strip()}


@router.get("/db/connections", summary="[BD 3] Connexions enregistrées du tenant")
async def db_connections(
    tenant_id: str,
    ident: Identity = Depends(identity),
):
    _ensure_tenant(ident, tenant_id)
    tid = normalize_tenant_id(tenant_id)
    return {"connections": await _db_call(db_manager.list_connections, tid)}


@router.get("/db/{connection_id}/schemas", summary="[BD 4] Schémas")
async def db_schemas(
    connection_id: str,
    tenant_id: str,
    ident: Identity = Depends(identity),
):
    _ensure_tenant(ident, tenant_id)
    tid = normalize_tenant_id(tenant_id)
    return {"schemas": await _db_call(db_manager.list_schemas, connection_id, tid)}


@router.get("/db/{connection_id}/tables", summary="[BD 5] Tables et vues d'un schéma")
async def db_tables(
    connection_id: str,
    tenant_id: str,
    schema: str,
    ident: Identity = Depends(identity),
):
    _ensure_tenant(ident, tenant_id)
    tid = normalize_tenant_id(tenant_id)
    return {
        "tables": await _db_call(db_manager.list_tables, connection_id, tid, schema)
    }


@router.get("/db/{connection_id}/columns", summary="[BD 6] Colonnes + lignes d'exemple")
async def db_columns(
    connection_id: str,
    tenant_id: str,
    schema: str,
    table: str,
    ident: Identity = Depends(identity),
):
    _ensure_tenant(ident, tenant_id)
    tid = normalize_tenant_id(tenant_id)
    return await _db_call(db_manager.list_columns, connection_id, tid, schema, table)


@router.post("/db/ingest", status_code=202, summary="[BD 7] Lire, formater et ingérer")
async def db_ingest(
    tenant_id: str = Form(..., description="Identifiant du tenant"),
    connection_id: str = Form(...),
    mapping: str = Form(..., description="JSON de correspondance colonnes → en-tête"),
    ident: Identity = Depends(identity),
    service=Depends(get_service),
):
    _ensure_tenant(ident, tenant_id)
    tenant_id = normalize_tenant_id(tenant_id)
    if not tenant_id:
        raise HTTPException(400, "tenant_id invalide.")
    try:
        mp = json.loads(mapping)
    except ValueError:
        raise HTTPException(400, "mapping doit être un JSON valide.")

    rows = await _db_call(db_manager.read_mapped_rows, connection_id, tenant_id, mp)
    blocks, skipped = db_manager.build_blocks(rows, mp, tenant_id)
    if not blocks:
        raise HTTPException(
            400, "Aucune ligne exploitable (colonnes vides ou rôles manquants)."
        )

    base = mp["base"]
    result = await _db_call(
        service.ingest_formatted_texts,
        texts=blocks,
        filename=f"db_{connection_id[:8]}_{base['schema']}_{base['table']}",
        tenant_id=tenant_id,
        # update_rbac=False,
    )
    return {
        "status": "success",
        "tenant_id": tenant_id,
        "rows_read": len(rows),
        "blocks_ingested": len(blocks),
        "rows_skipped": skipped,
        "ingestion": result,
    }


# -----------------
@router.post("/text", status_code=202)
def ingest_text(
    body: TextIn, ident: Identity = Depends(identity), service=Depends(get_service)
):
    cfg = get_cfg()
    tmp_dir = cfg.data_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=tmp_dir, delete=False, suffix=".md", mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(body.text)
        tmp_path = Path(tmp.name)
    try:
        ctx = IngestContext(
            tenant_id=ident.tenant_id, owner_id=ident.user_id, role=ident.role
        )
        return service.submit_upload(
            tmp_path=tmp_path, filename=f"{body.title}.md", ctx=ctx
        )
    except UnsupportedFile as e:
        raise HTTPException(415, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/documents")
def list_documents(ident: Identity = Depends(identity), service=Depends(get_service)):
    docs = service.list_documents(
        tenant_id=ident.tenant_id, user_id=ident.user_id, role=ident.role
    )
    return [_public(d) for d in docs]


@router.get("/documents/{doc_id}")
def get_document(
    doc_id: str, ident: Identity = Depends(identity), service=Depends(get_service)
):
    try:
        d = service.get_document(
            tenant_id=ident.tenant_id,
            doc_id=doc_id,
            user_id=ident.user_id,
            role=ident.role,
        )
    except ValueError:
        d = None
    if d is None:
        raise HTTPException(404, "Document introuvable")
    return _public(d)


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(
    doc_id: str, ident: Identity = Depends(identity), service=Depends(get_service)
):
    try:
        ok = service.delete_document(
            tenant_id=ident.tenant_id,
            doc_id=doc_id,
            user_id=ident.user_id,
            role=ident.role,
        )
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError:
        ok = False
    if not ok:
        raise HTTPException(404, "Document introuvable")


# --- Endpoints ---


@router.post(
    "/discover-start",
    summary="[Exploration 1/2] Démarrer une session d'exploration récursive d'URL",
    description="Initialise une session, scrappe la page de départ et retourne le premier niveau de liens à sélectionner.",
)
async def discover_start_endpoint(
    payload: StartDiscoveryRequest, ident: Identity = Depends(identity)
):
    _ensure_tenant(ident, payload.tenant_id)
    tenant_id = normalize_tenant_id(payload.tenant_id)
    cfg = get_cfg()
    session_id = payload.session_id or uuid.uuid4().hex
    _validate_discovery_session_id(session_id)

    session_dir = cfg.data_dir / "tmp" / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    cancel_file = _cancel_file(session_dir)
    cancel_file.unlink(missing_ok=True)

    text_file = session_dir / "accumulated_text.txt"
    state_file = session_dir / "state.json"

    try:
        accumulated_text, discovered_links, visited = await run_in_threadpool(
            crawl_and_accumulate_sync,
            urls_to_crawl=[payload.start_url],
            already_visited_urls=set(),
            cancel_check=cancel_file.exists,
        )

        # Enregistrement du texte initial
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(accumulated_text)

        # Enregistrement de l'état de session
        state = {
            "session_id": session_id,
            "tenant_id": tenant_id,
            "start_url": payload.start_url,
            "visited_urls": list(visited),
            "pages_count": len(visited),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f)

        return {
            "status": "in_progress",
            "session_id": session_id,
            "tenant_id": tenant_id,
            "current_page": payload.start_url,
            "pages_scraped_so_far": len(visited),
            "discovered_links": discovered_links,
            "message": "Sélectionnez les URLs à approfondir et appelez /discover-continue avec continue_exploration=true, ou mettez continue_exploration=false pour finaliser.",
        }

    except ScrapingBlockedException as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "scraping_blocked",
                "display_message_html": f'<i style="color:red;">L\'URL ({payload.start_url}) ne peut pas être scrappée : {e.reason}.</i>',
            },
        )
    except ScrapingCancelled:
        import shutil

        shutil.rmtree(session_dir, ignore_errors=True)
        return {
            "status": "cancelled",
            "session_id": session_id,
            "message": "Exploration annulée à la demande de l'utilisateur.",
        }


@router.post(
    "/discover-cancel/{session_id}",
    summary="Annuler une exploration ou une finalisation en cours",
)
async def discover_cancel_endpoint(
    session_id: str, ident: Identity = Depends(identity)
):
    _validate_discovery_session_id(session_id)
    session_dir = get_cfg().data_dir / "tmp" / "sessions" / session_id
    if not session_dir.is_dir():
        raise HTTPException(404, "Session d'exploration introuvable ou terminée.")
    state_file = session_dir / "state.json"
    if not state_file.is_file():
        raise HTTPException(404, "Session d'exploration introuvable ou terminée.")
    with state_file.open(encoding="utf-8") as state_handle:
        state = json.load(state_handle)
    _ensure_tenant(ident, state.get("tenant_id", ""))

    _cancel_file(session_dir).touch()
    return {
        "status": "cancellation_requested",
        "session_id": session_id,
        "message": "La demande d'arrêt a été transmise au traitement en cours.",
    }


@router.post(
    "/discover-continue",
    summary="[Exploration 2/2] Continuer l'exploration ou finaliser l'ingestion",
    description="Si continue_exploration=true, scrappe les liens choisis et renvoie les suivants. Si false, arrête le processus, rassemble tout le texte et procède à la conversion/ingestion LanceDB.",
)
async def discover_continue_endpoint(
    payload: ContinueDiscoveryRequest, ident: Identity = Depends(identity)
):
    _ensure_tenant(ident, payload.tenant_id)
    _validate_discovery_session_id(payload.session_id)
    tenant_id = normalize_tenant_id(payload.tenant_id)
    cfg = get_cfg()
    session_dir = cfg.data_dir / "tmp" / "sessions" / payload.session_id

    text_file = session_dir / "accumulated_text.txt"
    state_file = session_dir / "state.json"

    if not text_file.exists() or not state_file.exists():
        raise HTTPException(
            status_code=404, detail="Session d'exploration introuvable ou expirée."
        )

    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
    _ensure_tenant(ident, state.get("tenant_id", ""))

    visited_urls = set(state.get("visited_urls", []))
    cancel_file = _cancel_file(session_dir)
    cancel_file.unlink(missing_ok=True)
    cancelled = False

    try:
        # === CAS 1 : L'utilisateur souhaite continuer le scraping sur de nouvelles pages ===
        if payload.continue_exploration:
            if not payload.selected_urls:
                return {
                    "status": "in_progress",
                    "session_id": payload.session_id,
                    "message": "Aucune URL sélectionnée. Veuillez fournir des URLs dans 'selected_urls' ou mettre continue_exploration=false pour terminer.",
                    "discovered_links": [],
                }

            # Scraping synchrone du lot d'URLs sélectionnées
            new_text, new_discovered_links, updated_visited = await run_in_threadpool(
                crawl_and_accumulate_sync,
                urls_to_crawl=payload.selected_urls,
                already_visited_urls=visited_urls,
                cancel_check=cancel_file.exists,
            )

            # Append du nouveau texte scrappé
            with open(text_file, "a", encoding="utf-8") as f:
                f.write("\n\n" + new_text)

            # Mise à jour de l'état
            state["visited_urls"] = list(updated_visited)
            state["pages_count"] = len(updated_visited)
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f)
                print(
                    f"État de session mis à jour : {len(updated_visited)} pages visitées."
                )

            return {
                "status": "in_progress",
                "session_id": payload.session_id,
                "tenant_id": tenant_id,
                "pages_scraped_so_far": len(updated_visited),
                "discovered_links": new_discovered_links,
                "message": f"{len(payload.selected_urls)} nouvelle(s) page(s) ajoutée(s). Soumettez la sélection suivante ou continue_exploration=false pour terminer.",
            }

        # === CAS 2 : L'utilisateur répond NON (continue_exploration = False) -> FINALISATION ===
        else:
            print(
                "Finalisation de l'exploration et préparation à l'ingestion LanceDB..."
            )
            with open(text_file, "r", encoding="utf-8") as f:
                full_accumulated_text = f.read()

            if not full_accumulated_text.strip():
                raise HTTPException(
                    status_code=400, detail="Le contenu accumulé est vide."
                )

            print(f"Contenu accumulé : {len(full_accumulated_text)} caractères.")

            # Passage au convertisseur extensible standard (convert_and_reformat_document)
            formatted_chunks = []
            chunk_index = 1

            print(
                f"Conversion et reformattage du document accumulé pour le tenant '{tenant_id}'..."
            )

            formatted_chunks_iterator = convert_and_reformat_document(
                file_path=text_file,
                filename=f"session_{payload.session_id}.txt",
                tenant_id=tenant_id,
                pages_per_chunk=1,
                cancel_check=cancel_file.exists,
            )
            while True:
                if cancel_file.exists():
                    cancelled = True
                    return {
                        "status": "cancelled",
                        "session_id": payload.session_id,
                        "message": "Finalisation interrompue. Vous pouvez la relancer pour reprendre le traitement.",
                        "total_chunks_processed": len(formatted_chunks),
                    }

                formatted_block = await run_in_threadpool(
                    _next_chunk_or_none, formatted_chunks_iterator
                )
                if formatted_block is None:
                    break

                if cancel_file.exists():
                    cancelled = True
                    return {
                        "status": "cancelled",
                        "session_id": payload.session_id,
                        "message": "Finalisation interrompue. Vous pouvez la relancer pour reprendre le traitement.",
                        "total_chunks_processed": len(formatted_chunks),
                    }
                formatted_chunks.append(
                    {"chunk_id": chunk_index, "formatted_text": formatted_block}
                )
                chunk_index += 1
                print(
                    f"Chunk {chunk_index} formaté : {len(formatted_block)} caractères."
                )

            if cancel_file.exists():
                cancelled = True
                return {
                    "status": "cancelled",
                    "session_id": payload.session_id,
                    "message": "Finalisation interrompue. Vous pouvez la relancer pour reprendre le traitement.",
                    "total_chunks_processed": len(formatted_chunks),
                }

            print(f"Total de {len(formatted_chunks)} chunks formatés pour l'ingestion.")
            """
            # Ingestion optionnelle dans LanceDB
            ingestion_result = None
            if payload.ingest and formatted_chunks:
                service = get_default_service()
                ingestion_result = await run_in_threadpool(
                    service.ingest_formatted_texts,
                    texts=[c["formatted_text"] for c in formatted_chunks],
                    filename=f"URL_Session_{payload.session_id}",
                    tenant_id=tenant_id,
                )
            """

            return {
                "status": "completed",
                "session_id": payload.session_id,
                "tenant_id": tenant_id,
                "total_pages_scraped": state.get("pages_count", 1),
                "total_chunks_processed": len(formatted_chunks),
                "chunks": formatted_chunks,
                # "ingestion": ingestion_result,
            }

    except ScrapingCancelled:
        cancelled = True
        return {
            "status": "cancelled",
            "session_id": payload.session_id,
            "message": "Scraping interrompu à la demande de l'utilisateur. Vous pouvez continuer l'exploration.",
        }
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(f"Erreur pendant l'exploration : {str(e)}\n{stack_trace}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": f"Erreur pendant l'exploration : {str(e)}",
                "details": {"traceback": stack_trace.splitlines()[-5:]},
            },
        )
    finally:
        print(
            f"Session {payload.session_id} terminée. Nettoyage des fichiers temporaires..."
        )
        # Nettoyage si la session est terminée
        if not payload.continue_exploration and not cancelled and session_dir.exists():
            import shutil

            shutil.rmtree(session_dir, ignore_errors=True)
        elif cancelled:
            cancel_file.unlink(missing_ok=True)


# ---------------------------- SEMNANTIQUE TRAITEMENTS IN CUBE -----------------------


@router.get(
    "/semantic-models/connection",
    summary="Connexion utilisée par la couche sémantique du tenant",
)
async def semantic_models_connection(ident: Identity = Depends(identity)):
    tenant_id = normalize_tenant_id(ident.tenant_id)
    connection = await _db_call(db_manager.get_latest_connection_summary, tenant_id)
    return {"tenant_id": tenant_id, "connection": connection}


@router.post(
    "/semantic-models/index",
    status_code=200,
    summary="Installer et indexer les modèles YAML Cube du tenant, il prends en charge un tableau d'éléments",
)
async def semantic_models_index(
    files: List[UploadFile] = File(...),
    ident: Identity = Depends(identity),
    service=Depends(get_service),
):
    tenant_id = normalize_tenant_id(ident.tenant_id)
    if not tenant_id:
        raise HTTPException(400, "Identifiant tenant invalide.")
    if not files:
        raise HTTPException(400, "Chargez au moins un fichier YAML.")

    seen_filenames = set()
    for uploaded_file in files:
        filename = uploaded_file.filename or ""
        if (
            not filename
            or Path(filename).name != filename
            or Path(filename).suffix.lower() not in {".yaml", ".yml"}
        ):
            raise HTTPException(400, f"Nom de fichier YAML invalide : {filename!r}.")
        filename_key = filename.casefold()
        if filename_key in seen_filenames:
            raise HTTPException(
                400, f"Le fichier {filename!r} est fourni plusieurs fois."
            )
        seen_filenames.add(filename_key)

    db_config = await _db_call(db_manager.get_latest_connection_config, tenant_id)
    if not db_config:
        raise HTTPException(
            409,
            "Aucune connexion analytique n’est enregistrée pour cette entreprise. "
            "Configurez-la depuis le profil administrateur.",
        )

    indexed_files = []
    for uploaded_file in files:
        filename = uploaded_file.filename or ""
        try:
            await uploaded_file.seek(0)
            indexed_files.append(
                await run_in_threadpool(
                    service.index_semantic_file,
                    file_obj=uploaded_file.file,
                    filename=filename,
                    tenant_id=tenant_id,
                )
            )
        except SemanticFileTooLargeError as exc:
            raise HTTPException(413, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.exception("Erreur d’indexation LanceDB du fichier %s", filename)
            raise HTTPException(
                502, f"Erreur d’indexation dans LanceDB pour {filename!r} : {exc}"
            ) from exc

    return {
        "status": "indexed",
        "tenant_id": tenant_id,
        "file_indexed": len(indexed_files),
        "files": indexed_files,
    }


@router.post(
    "/semantic-models/cube-connect",
    summary="Envoyer la connexion BD du tenant à Cube et vérifier",
)
async def semantic_models_cube_connect(ident: Identity = Depends(identity)):
    tenant_id = normalize_tenant_id(ident.tenant_id)
    if not tenant_id:
        raise HTTPException(400, "Identifiant tenant invalide.")

    db_config = await _db_call(db_manager.get_latest_connection_config, tenant_id)
    if not db_config:
        raise HTTPException(
            409,
            "Aucune connexion analytique n’est enregistrée pour cette entreprise. "
            "Configurez-la depuis le profil administrateur.",
        )

    company_context = {
        "tenantId": tenant_id,
        "type": db_config.get("db_type") or db_config.get("type") or "postgres",
        "host": db_config.get("host"),
        "port": db_config.get("port"),
        "database": db_config.get("database"),
        "user": db_config.get("username") or db_config.get("user"),
        "password": db_config.get("password"),
    }

    print("db config", company_context)

    missing = [
        k for k in ("host", "database", "user", "password") if not company_context[k]
    ]
    if missing:
        raise HTTPException(
            500, f"Configuration de connexion incomplète : {', '.join(missing)}."
        )

    dev = await push_cube_dev_context(company_context)
    result = await test_cube_connection(company_context)
    return {**result, "dev_ui_ready": dev.get("ok", False)}
