# Fichier : app/services/redis_client.py
import logging

import redis.asyncio as redis

log = logging.getLogger("redis.client")

# Connexion asynchrone pointant vers le service docker "redis" ou "localhost:6379"
REDIS_URL = "redis://localhost:6379/0"

_redis_client: redis.Redis | None = None


async def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            await _redis_client.ping()
            log.info("[Redis] Connexion établie avec succès.")
        except Exception as e:
            log.error(f"[Redis] Échec de la connexion : {e}")
            raise e
    return _redis_client


async def close_redis_client():
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        log.info("[Redis] Connexion fermée.")
