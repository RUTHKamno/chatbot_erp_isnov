# Fichier : app/api/sma_api.py
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.graph.workflow import sma_graph

log = logging.getLogger(__name__)

router = APIRouter(tags=["Guide ERP & Chat"])

# Singleton pour le schéma de sécurité (évite la règle Ruff B008 sur le second argument)
security_scheme = HTTPBearer(auto_error=False)


class ChatRequest(BaseModel):
    user_id: str = "user_001"
    raw_query: str


class ChatResponse(BaseModel):
    status: str
    session_id: str
    intent_detecte: str | None = None
    answer: str | None = None
    suggestions: list[dict[str, str]] | None = None


def decode_jwt_claims(authorization: str | None) -> dict[str, Any]:
    """Extrait les claims du JWT s'il est présent, sinon fournit les défauts."""
    if not authorization:
        return {"tenant_id": "default", "sub": "user_001", "role": "all"}
    try:
        import jwt

        from app.config import get_settings

        settings = get_settings()
        token = authorization.removeprefix("Bearer ").strip()
        return jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[getattr(settings, "jwt_algorithm", "HS256")],
        )
    except Exception:
        return {"tenant_id": "default", "sub": "user_001", "role": "all"}


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
