# Fichier : app/api/sma_api.py
import logging
import os
import shutil
import tempfile
import uuid
from typing import Any

import jwt
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.core.audio_model_def import (
    TranscriptionError,
    transcribe_audio_bytes,
)
from app.graph.workflow import sma_graph

log = logging.getLogger(__name__)

router = APIRouter(tags=["Guide ERP & Chat"])

from fastapi.responses import Response
from langgraph.types import Command

from app.core.tts_model_def import generate_speech_bytes

# Singleton pour le schéma de sécurité (évite la règle Ruff B008 sur le second argument)
security_scheme = HTTPBearer(auto_error=False)


# class ChatRequest(BaseModel):
#    user_id: str = "user_001"
#    raw_query: str


class ChatRequest(BaseModel):
    user_id: str = "user_001"
    raw_query: str | None = None
    input_mode: str = "text"  # "text" ou "voice"
    validation_status: str | None = None  # "validated" ou "rejected"
    voice_retry_count: int = 0


class ChatResponse(BaseModel):
    status: str
    session_id: str
    intent_detecte: str | None = None
    answer: str | None = None
    chart_config: dict[str, Any] | None = None
    data: list[dict[str, Any]] | None = None
    suggestions: list[dict[str, str]] | None = None


def decode_jwt_claims(authorization: str | None) -> dict[str, Any]:
    """Decode a required bearer token; never grant a default anonymous identity."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentification requise.")
    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token.strip():
        raise HTTPException(status_code=401, detail="Authentification Bearer requise.")
    try:
        from app.config import get_settings

        settings = get_settings()
        claims = jwt.decode(
            token.strip(),
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        if not all(claims.get(key) for key in ("sub", "tenant_id", "role")):
            raise HTTPException(status_code=403, detail="Claims d’identité incomplets.")
        return claims
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=401, detail="Token invalide ou expiré."
        ) from exc


"""
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    auth: HTTPAuthorizationCredentials | None = Depends(security_scheme),  # noqa: B008
):
    session_id = str(uuid.uuid4())

    # Extraction du token depuis le bouton Authorize de Swagger ou Header
    token = auth.credentials if auth else None
    claims = decode_jwt_claims(f"Bearer {token}" if token else None)

    # Initialisation de l'état LangGraph
    initial_state = {
        "user_id": claims.get("sub", request.user_id),
        "session_id": session_id,
        "raw_query": request.raw_query,
        "messages": [("user", request.raw_query)],
        "jwt_claims": claims,
    }

    # Exécution du graphe
    result_state = await sma_graph.ainvoke(initial_state)

    return ChatResponse(
        status="success",
        session_id=session_id,
        intent_detecte=result_state.get("intent"),
        answer=result_state.get("answer"),
        suggestions=result_state.get("suggestions", []),
    )
"""


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    user_id: str = Form("user_001"),
    raw_query: str | None = Form(None),
    input_mode: str = Form("text"),
    session_id: str | None = Form(
        None
    ),  # Récupéré si on est dans une session existante en pause
    user_action: str | None = Form(None),  # "validate" ou "reject"
    voice_retry_count: int = Form(0),
    audio_file: UploadFile | None = File(None),
    auth: HTTPAuthorizationCredentials | None = Depends(security_scheme),
):
    # Si pas de session_id, on en crée un nouveau (Thread ID pour LangGraph)
    current_session_id = session_id or str(uuid.uuid4())
    token = auth.credentials if auth else None
    claims = decode_jwt_claims(f"Bearer {token}" if token else None)
    config = {"configurable": {"thread_id": current_session_id}}

    temp_audio_path = None
    if input_mode == "voice" and audio_file and not session_id:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(audio_file.file, tmp)
            temp_audio_path = tmp.name

    try:
        # CAS 1 : Reprise d'une interruption (l'utilisateur a cliqué sur Valider ou Refuser sur le front)
        if session_id and user_action:
            resume_payload = {
                "action": user_action,
                "query": raw_query,  # La requête validée ou corrigée
            }
            # Utilisation de Command pour reprendre l'exécution du graphe là où il s'était arrêté
            result_state = await sma_graph.ainvoke(
                Command(resume=resume_payload), config
            )

        # CAS 2 : Premier lancement d'une requête (Texte ou Audio initial)
        else:
            initial_state = {
                "user_id": claims.get("sub", user_id),
                "session_id": current_session_id,
                "raw_query": raw_query or "",
                "input_mode": input_mode,
                "audio_file_path": temp_audio_path,
                "voice_retry_count": voice_retry_count,
                "messages": [("user", raw_query)] if raw_query else [],
                "jwt_claims": claims,
            }
            result_state = await sma_graph.ainvoke(initial_state, config)

        # Vérifier si le graphe est en pause (interrompu par interrupt())
        # LangGraph renvoie un état spécial ou l'on peut vérifier les valeurs en attente
        # (Dans les versions récentes, si une interruption est active, l'exécution s'arrête et retourne la valeur de l'interrupt)

    except Exception as e:
        log.error(f"Erreur d'exécution du graphe : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

    return ChatResponse(
        status="success",
        session_id=current_session_id,
        intent_detecte=result_state.get("intent"),
        answer=result_state.get("answer"),
        chart_config=result_state.get("chart_config"),
        data=(result_state.get("sql_result") or {}).get("data", []),
        suggestions=result_state.get("suggestions", []),
    )


"""
@router.on_event("startup")
def startup_event():
    # Nettoyer les vieux fichiers temporaires qui auraient pu stagner au démarrage
    purge_old_temp_audio(max_age_seconds=60)
"""


@router.post(
    "/transcribe", summary="Transcrire un fichier audio directement depuis la mémoire"
)
async def api_transcribe_audio(
    file: UploadFile = File(..., description="Fichier audio (wav, mp3, m4a, etc.)"),
    language: str = Form(None, description="Code langue optionnel ('fr', 'en')."),
):
    allowed_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté. Autorisé : {list(allowed_extensions)}",
        )

    try:
        # On lit les octets directement issus de la requête HTTP
        audio_bytes = await file.read()

        # Transcription directe sans jamais toucher au disque dur
        result = await transcribe_audio_bytes(
            audio_bytes=audio_bytes, language=language if language else None
        )

        return {
            "text": result.text,
            "language": result.language,
            "language_probability": round(result.language_probability, 4),
            "duration_seconds": round(result.duration, 2),
        }

    except TranscriptionError as te:
        raise HTTPException(status_code=422, detail=str(te))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur interne du serveur : {str(e)}"
        )


@router.post(
    "/text-to-speech", summary="Convertit un texte en audio WAV selon la langue"
)
async def api_text_to_speech(
    text: str = Form(..., description="Le texte à convertir en audio"),
    language: str = Form(
        "fr", description="Langue détectée pour la voix ('fr' ou 'en')"
    ),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    audio_bytes = generate_speech_bytes(text, language=language)

    if not audio_bytes:
        raise HTTPException(
            status_code=500, detail="Erreur lors de la génération audio."
        )

    return Response(content=audio_bytes, media_type="audio/wav")
