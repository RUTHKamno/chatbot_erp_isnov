"""API d'ingestion. tenant_id / sub / role viennent EXCLUSIVEMENT du JWT (jamais du corps de la requête)."""

import json
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import List
import uuid  # noqa: UP035

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
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from sympy import false

from app.ingestion.converter import convert_and_reformat_document
from app.ingestion.models import IngestContext
from app.ingestion.pipeline import get_default_service, normalize_tenant_id
from app.ingestion.security import UnsupportedFile
from app.ingestion.settings import get_cfg
from app.ingestion.web_scrapy import (
    ScrapingBlockedException,
    crawl_and_accumulate_sync,
    crawl_multiple_selected_urls_sync,
    extract_markdown_links,
    run_crawl4ai_sync,
)

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Logger dédié pour les tests et erreurs de conversion
logger = logging.getLogger("ingestion.test_jalon1")
logger.setLevel(logging.INFO)


# >>> À ADAPTER : remplace cette fonction par ton décodage JWT existant (elle doit renvoyer les claims) <<<
def current_claims(authorization: str = Header(...)) -> dict:
    import jwt  # PyJWT

    from app.config import get_settings

    s = get_settings()
    token = authorization.removeprefix("Bearer ").strip()
    try:
        return jwt.decode(
            token, s.jwt_secret, algorithms=[getattr(s, "jwt_algorithm", "HS256")]
        )
    except Exception:
        raise HTTPException(401, "Token invalide ou expiré")


@dataclass
class Identity:
    tenant_id: str
    user_id: str
    role: str


class StartDiscoveryRequest(BaseModel):
    start_url: str = Field(..., description="URL initiale à explorer")
    tenant_id: str = Field("dolibarr", description="Identifiant du tenant")


class ContinueDiscoveryRequest(BaseModel):
    session_id: str = Field(
        ..., description="Identifiant de la session d'exploration en cours"
    )
    tenant_id: str = Field("dolibarr", description="Identifiant du tenant")
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
    if (
        role not in get_cfg().admin_roles
    ):  # le JWT sert ici à vérifier le DROIT d'appeler l'API d'ingestion
        raise HTTPException(403, "Droit d'ingestion requis (rôle administrateur).")
    return Identity(tenant, sub, role)


def get_service():
    from app.ingestion.pipeline import get_default_service

    return get_default_service()


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


@router.post("/upload", status_code=202, summary="Ingérer un document ERP")
async def upload(
    file: UploadFile = File(...),
    ident: Identity = Depends(identity),
    service=Depends(get_service),
):
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
                tenant_name=ident.tenant_id,
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
            tenant_id=ident.tenant_id,
        )

        return {
            "status": "success",
            "filename": file.filename,
            "tenant_id": ident.tenant_id,
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
async def discover_start_endpoint(payload: StartDiscoveryRequest):
    tenant_id = normalize_tenant_id(payload.tenant_id)
    cfg = get_cfg()
    session_id = uuid.uuid4().hex

    session_dir = cfg.data_dir / "tmp" / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    text_file = session_dir / "accumulated_text.txt"
    state_file = session_dir / "state.json"

    try:
        accumulated_text, discovered_links, visited = await run_in_threadpool(
            crawl_and_accumulate_sync,
            urls_to_crawl=[payload.start_url],
            already_visited_urls=set(),
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


@router.post(
    "/discover-continue",
    summary="[Exploration 2/2] Continuer l'exploration ou finaliser l'ingestion",
    description="Si continue_exploration=true, scrappe les liens choisis et renvoie les suivants. Si false, arrête le processus, rassemble tout le texte et procède à la conversion/ingestion LanceDB.",
)
async def discover_continue_endpoint(payload: ContinueDiscoveryRequest):
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

    visited_urls = set(state.get("visited_urls", []))

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

            for formatted_block in convert_and_reformat_document(
                file_path=text_file,
                filename=f"session_{payload.session_id}.txt",
                tenant_id=tenant_id,
                pages_per_chunk=1,
            ):
                formatted_chunks.append(
                    {"chunk_id": chunk_index, "formatted_text": formatted_block}
                )
                chunk_index += 1
                print(
                    f"Chunk {chunk_index} formaté : {len(formatted_block)} caractères."
                )

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
        if not payload.continue_exploration and session_dir.exists():
            import shutil

            shutil.rmtree(session_dir, ignore_errors=True)
