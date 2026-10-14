# Fichier : app/services/cache_service.py
import hashlib
import json
import logging

from app.core.redis_client import get_redis_client

log = logging.getLogger("service.cache")


class RedisCacheService:
    @staticmethod
    def _hash_query(query: str) -> str:
        """Génère une clé de hachage unique pour le cache exact."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @classmethod
    async def get_cached_answer(cls, query: str) -> dict | None:
        """
        Vérifie d'abord le cache exact (via hash), puis sémantique si configuré.
        Retourne la réponse si trouvée, sinon None.
        """
        print(f"[Cache] Vérification du cache pour la requête : '{query}'")
        try:
            client = await get_redis_client()

            # 1. Vérification du cache exact
            exact_key = f"cache:exact:{cls._hash_query(query)}"
            cached_data = await client.get(exact_key)

            if cached_data:
                print(f"[Cache] Cache HIT exact pour : '{query}'")
                log.info(f"[Cache] Cache HIT exact pour : '{query}'")
                return json.loads(cached_data)

            # 2. (Optionnel / Extensible) Vérification sémantique
            # On pourrait parcourir les clés de cache récentes ou utiliser RediSearch
            # Pour l'instant, le cache exact assure une ultra-rapidité sur les requêtes répétées.
            print(f"[Cache] Cache MISS pour : '{query}'")
            log.info(f"[Cache] Cache MISS pour : '{query}'")
            return None

        except Exception as e:
            print(f"[Cache Error] Erreur lors de la lecture du cache : {e}")
            log.error(f"[Cache Error] Erreur lors de la lecture du cache : {e}")
            return None

    @classmethod
    async def set_cached_answer(cls, query: str, answer_data: dict, ttl: int = 3600):
        """
        Enregistre une réponse dans le cache Redis avec un TTL (par défaut 1 heure).
        """
        print("[Cache] Mise en cache de la réponse...")
        try:
            client = await get_redis_client()
            exact_key = f"cache:exact:{cls._hash_query(query)}"

            # Stockage de la réponse sérialisée en JSON
            await client.setex(exact_key, ttl, json.dumps(answer_data))
            print(f"[Cache] Réponse mise en cache pour : '{query}' (TTL: {ttl}s)")
            log.info(f"[Cache] Réponse mise en cache pour : '{query}' (TTL: {ttl}s)")
        except Exception as e:
            log.error(f"[Cache Error] Erreur lors de l'écriture en cache : {e}")
