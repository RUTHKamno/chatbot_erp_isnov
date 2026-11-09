# Fichier : app/services/redis_client.py
import logging

import redis.asyncio as redis

from app.config import get_settings

log = logging.getLogger("redis.client")

_redis_client: redis.Redis | None = None


async def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(
                get_settings().redis_url, decode_responses=True
            )
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
