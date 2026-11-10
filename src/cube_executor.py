import httpx
import asyncio
import json
import os
import time
import jwt
from typing import Dict, Any, List

CUBE_API_URL = os.environ.get("CUBE_API_URL", "http://localhost:4000").rstrip("/")
if not CUBE_API_URL.endswith("/cubejs-api"):
    CUBE_API_URL += "/cubejs-api"

CUBE_API_SECRET = os.environ.get(
    "CUBE_API_SECRET", 
    "3dfea42a566996f37c5fdbe97f14749086a40e59fcd03f96a688782828bd5128"
)

# Contexte par défaut pour les tests locaux (pointe vers ta base PostgreSQL distante)
DEFAULT_DB_CONTEXT = {
    "tenantId": "isnov",
    "type": "postgres",
    "host": "188.68.36.158",
    "port": 6425,
    "database": "bi_dwh_std_r2501",
    "user": "query_user",
    "password": "query_user",
}

def generate_cube_token(company_db_context: dict = None) -> str:
    """Génère un jeton JWT sécurisé lu par Cube.js pour l'authentification et le multi-tenant."""
    ctx = company_db_context or DEFAULT_DB_CONTEXT
    
    # 🔍 PRINT DE DIAGNOSTIC DES CREDENTIALS
    print(f"\n🔍 [CUBE CREDENTIALS] Tentative de connexion sémantique -> "
          f"Hôte: {ctx.get('host')}:{ctx.get('port')} | "
          f"Base: {ctx.get('database')} | "
          f"User: {ctx.get('user')} | "
          f"Tenant: {ctx.get('tenantId')}")

    payload = {
        "tenantId": ctx.get("tenantId", "default"),
        "companyDbType": ctx.get("type", "postgres"),
        "companyDbHost": ctx.get("host", "188.68.36.158"),
        "companyDbPort": ctx.get("port", 6425),
        "companyDbName": ctx.get("database", "bi_dwh_std_r2501"),
        "companyDbUser": ctx.get("user", "query_user"),
        "companyDbPassword": ctx.get("password", "query_user"),
        "exp": int(time.time()) + 3600,
    }
    return jwt.encode(payload, CUBE_API_SECRET, algorithm="HS256")

async def execute_single_query(item: Dict[str, Any]) -> Dict[str, Any]:
    """Exécute une seule requête Cube.js de manière asynchrone via l'API /v1/load."""
    title = item.get("component_title", "Sans titre")
    ctype = item.get("component_type", "unknown")
    query = item.get("cube_query")
    status = item.get("status", "failed")

    if status == "failed" or not query:
        return {
            "component_title": title,
            "component_type": ctype,
            "cube_query": query,
            "status": "failed",
            "error": item.get("error", "Requête invalide ou non générée"),
            "data": []
        }

    token = generate_cube_token()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    payload = {"query": query}
    print(f"🚀 [CUBE EXECUTOR] Envoi de la requête pour '{title}'...")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{CUBE_API_URL}/v1/load", 
                json=payload, 
                headers=headers
            )

        if response.status_code == 200:
            res_json = response.json()
            data = res_json.get("data", [])
            print(f"✅ [CUBE SUCCESS] '{title}' -> {len(data)} ligne(s) récupérée(s).")
            return {
                "component_title": title,
                "component_type": ctype,
                "cube_query": query,
                "status": "success",
                "data": data,
                "error": None
            }
        else:
            err_msg = f"HTTP {response.status_code}: {response.text}"
            print(f"❌ [CUBE ERROR] '{title}' -> {err_msg}")
            return {
                "component_title": title,
                "component_type": ctype,
                "cube_query": query,
                "status": "error",
                "data": [],
                "error": err_msg
            }
    except Exception as e:
        err_msg = str(e)
        print(f"❌ [CUBE EXCEPTION] '{title}' -> {err_msg}")
        return {
            "component_title": title,
            "component_type": ctype,
            "cube_query": query,
            "status": "error",
            "data": [],
            "error": err_msg
        }

async def execute_and_buffer_all(queries_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exécute toutes les requêtes en parallèle et accumule les résultats dans le Buffer."""
    print(f"\n📦 [BUFFER START] Début de l'exécution pour {len(queries_results)} composant(s)...")
    tasks = [execute_single_query(item) for item in queries_results]
    buffered_results = await asyncio.gather(*tasks)
    print(f"📦 [BUFFER COMPLETE] {len(buffered_results)} résultats accumulés dans la zone tampon.\n")
    return buffered_results

def run_cube_buffer(queries_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Helper synchrone pour appeler l'exécuteur depuis Streamlit."""
    return asyncio.run(execute_and_buffer_all(queries_results))