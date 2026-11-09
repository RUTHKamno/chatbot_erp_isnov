import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import jwt

logger = logging.getLogger(__name__)

CUBE_API_URL = os.getenv("CUBE_API_URL", "http://localhost:4000")
#CUBE_API_URL = "http://localhost:4000/cubejs-api/v1"
CUBE_API_SECRET = os.getenv("CUBE_API_SECRET", "")
print(f"DEBUG URL CUBE : [{CUBE_API_URL}]")


def _cube_load_url() -> str:
    base_url = CUBE_API_URL.rstrip("/")
    if base_url.endswith("/cubejs-api/v1"):
        return f"{base_url}/load"
    return f"{base_url}/cubejs-api/v1/load"


def _cube_meta_url() -> str:
    base_url = CUBE_API_URL.rstrip("/")
    if base_url.endswith("/cubejs-api/v1"):
        return f"{base_url}/meta"
    return f"{base_url}/cubejs-api/v1/meta"


print(f"DEBUG URLssssssss CUBE : [{_cube_load_url()}, {_cube_meta_url()}]")


def _cube_auth_headers(tenant_id: str, db_config: dict[str, Any]) -> dict[str, str]:
    if not CUBE_API_SECRET:
        raise RuntimeError("CUBE_API_SECRET n’est pas configurée.")
    if not db_config:
        raise ValueError("Aucune connexion analytique n’est configurée pour ce tenant.")

    from app.ingestion.pipeline import normalize_tenant_id

    security_context = {
        "tenantId": normalize_tenant_id(tenant_id)[:100],
        "companyDbName": db_config["database"],
        "companyDbType": db_config["type"],
        "companyDbHost": db_config["host"],
        "companyDbPort": str(db_config["port"]),
        "companyDbUser": db_config["user"],
        "companyDbPassword": db_config["password"],
        "exp": datetime.now(UTC) + timedelta(minutes=5),
    }
    token = jwt.encode(security_context, CUBE_API_SECRET, algorithm="HS256")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    return headers


async def fetch_cube_metadata(
    tenant_id: str, db_config: dict[str, Any]
) -> dict[str, Any]:
    """Ask Cube to load the tenant schemas using credentials held by the API."""
    logger.info("[Cube Client] Chargement des métadonnées Cube pour %s", tenant_id)
    async with httpx.AsyncClient(timeout=3600.0) as client:
        response = await client.get(
            _cube_meta_url(),
            headers=_cube_auth_headers(tenant_id, db_config),
        )
        if response.status_code == 200:
            return response.json()
        raise RuntimeError(
            f"Erreur API Cube metadata ({response.status_code}) : {response.text}"
        )


async def execute_cube_query(
    tenant_id: str, cube_query: dict[str, Any], db_config: dict[str, Any]
) -> dict[str, Any]:
    """Execute a Cube query using the encrypted-at-rest tenant connection settings."""

    logger.info("[Cube Client] Interrogation de Cube pour le tenant : %s", tenant_id)

    async with httpx.AsyncClient(timeout=3600.0) as client:
        response = await client.post(
            _cube_load_url(),
            json={"query": cube_query},
            headers=_cube_auth_headers(tenant_id, db_config),
        )
        if response.status_code == 200:
            return response.json()
        raise RuntimeError(
            f"Erreur API Cube ({response.status_code}) : {response.text}"
        )
