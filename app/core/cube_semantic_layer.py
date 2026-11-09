import asyncio
import os
import time

import httpx
import jwt  # Fourni par le package PyJWT de votre requirements.txt
from fastapi import HTTPException, status
from pydantic import BaseModel

CUBE_API_URL = os.getenv("CUBE_API_URL", "http://localhost:4000").rstrip("/")
if not CUBE_API_URL.endswith("/cubejs-api"):
    CUBE_API_URL += "/cubejs-api"
CUBE_API_SECRET = os.getenv("CUBE_API_SECRET")
CUBE_DEV_CONTEXT_URL = os.getenv("CUBE_DEV_CONTEXT_URL")


class CubeConnectRequest(BaseModel):
    tenant_id: str


def generate_cube_token(company_db_context: dict) -> str:
    """
    Génère un jeton JWT sécurisé contenant le contexte de la base de données
    de l'entreprise connectée. Ce jeton sera lu par cube.js.
    """
    payload = {
        # Les informations de connexion dynamiques (Option A)
        "tenantId": company_db_context.get(
            "tenantId", "default"
        ),  # <-- INDISPENSABLE pour les fichiers YAML
        "companyDbType": company_db_context.get(
            "type", "postgres"
        ),  # postgres, mysql, etc.
        "companyDbHost": company_db_context.get("host"),
        "companyDbPort": company_db_context.get("port", 5432),
        "companyDbName": company_db_context.get("database"),
        "companyDbUser": company_db_context.get("user"),
        "companyDbPassword": company_db_context.get("password"),
        # Sécurité JWT standard : expiration dans 1 heure
        "exp": int(time.time()) + 3600,
    }

    # Signature du token avec l'algorithme HS256
    return jwt.encode(payload, CUBE_API_SECRET, algorithm="HS256")


async def query_cube_semantic_layer(company_db_context: dict, cube_query: dict) -> dict:
    """
    Envoie une requête sémantique sécurisée à Cube pour récupérer
    les données de l'entreprise sans risque de fuite.
    """
    # 1. Génération du badge d'accès multi-tenant
    token = generate_cube_token(company_db_context)

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # 2. Appel asynchrone vers le conteneur Cube
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CUBE_API_URL}/v1/load",
                json={"query": cube_query},
                headers=headers,
                timeout=30.0,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Erreur Cube: {response.text}",
                )

            return response.json()

        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Impossible de joindre la couche sémantique Cube: {exc}",
            )


async def test_cube_connection(company_db_context: dict) -> dict:
    """
    Envoie le contexte BD à Cube (via le JWT) et vérifie que :
    - le token est accepté et les modèles YAML du tenant se chargent (/v1/meta)
    - la connexion à la base de l'entreprise fonctionne (requête limit 1)
    """
    token = generate_cube_token(company_db_context)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        try:
            print(f"{CUBE_API_URL}/v1/meta")
            meta = await client.get(
                f"{CUBE_API_URL}/v1/meta", headers=headers, timeout=30.0
            )
            if meta.status_code != 200:
                raise HTTPException(
                    status_code=meta.status_code,
                    detail=f"Erreur Cube (meta): {meta.text}",
                )

            print("cubes fnding", meta)

            cubes = meta.json().get("cubes", [])
            measure = next(
                (m["name"] for c in cubes for m in c.get("measures", [])), None
            )
            if measure is None:
                return {"status": "token_ok", "cubes": len(cubes), "db_checked": False}

            # Cette requête force cube.js à appeler driverFactory => teste la BD
            for _ in range(10):
                load = await client.post(
                    f"{CUBE_API_URL}/v1/load",
                    json={"query": {"measures": [measure], "limit": 1}},
                    headers=headers,
                    timeout=30.0,
                )
                body = load.json()
                if body.get("error") == "Continue wait":
                    await asyncio.sleep(1)
                    continue
                if load.status_code != 200 or "error" in body:
                    raise HTTPException(
                        status_code=(
                            load.status_code if load.status_code != 200 else 400
                        ),
                        detail=f"Erreur Cube (load): {body.get('error', load.text)}",
                    )
                return {"status": "ok", "cubes": len(cubes), "db_checked": True}

            raise HTTPException(status_code=504, detail="Cube n'a pas répondu à temps.")

        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Impossible de joindre la couche sémantique Cube: {exc}",
            )


async def push_cube_dev_context(company_db_context: dict) -> dict:
    """Envoie la connexion à Cube pour que l'interface (Data Model) l'utilise."""
    token = generate_cube_token(company_db_context)
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                CUBE_DEV_CONTEXT_URL,
                headers={"Authorization": f"Bearer {token}"},
                timeout=15.0,
            )
        except httpx.RequestError as exc:
            print(f"push_cube_dev_context: échec vers {CUBE_DEV_CONTEXT_URL}: {exc}")
            return {"ok": False}
    if r.status_code == 404:  # route absente (mode production)
        return {"ok": False}
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"Erreur Cube: {r.text}")
    return r.json()
