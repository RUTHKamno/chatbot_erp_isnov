# Fichier : main.py
from fastapi import FastAPI

# Importation des routeurs
from app.api.ingestion import router as ingestion_router
from app.api.sma_api import router as sma_api_router
from app.config import get_settings
from app.core.redis_client import close_redis_client, get_redis_client

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="API du système multi-agents basé sur LangGraph",
    version="1.0.0",
)


# Événements de cycle de vie (Lifecycle) pour Redis
@app.on_event("startup")
async def startup_event():
    try:
        await get_redis_client()
    except Exception:
        print("⚠️ Avertissement : Impossible de se connecter à Redis au démarrage.")


@app.on_event("shutdown")
async def shutdown_event():
    await close_redis_client()


# Enregistrement des endpoints
app.include_router(ingestion_router)
app.include_router(sma_api_router)


@app.get("/health")
def health_check():
    return {"status": "ok", "app_env": settings.app_env}


@app.get("/health/redis")
async def redis_health_check():
    """Route dédiée pour tester si la connexion à Redis passe correctement."""
    try:
        client = await get_redis_client()
        pong = await client.ping()
        return {"redis_status": "connected", "ping": pong}
    except Exception as e:
        return {"redis_status": "error", "details": str(e)}
