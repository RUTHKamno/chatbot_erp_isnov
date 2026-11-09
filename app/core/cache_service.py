# Fichier : app/services/cache_service.py
import hashlib
import json
import logging
import re
import unicodedata

from app.core.redis_client import get_redis_client

log = logging.getLogger("service.cache")


def normalize_query(query: str) -> str:
    """
    Normalise une requête pour uniformiser le cache :
    - Minuscules
    - Suppression des accents (ex: 'véux' -> 'veux')
    - Suppression de la ponctuation
    - Compression des espaces multiples
    """
    if not query:
        return ""

    # 1. Passage en minuscules
    text = query.lower()

    # 2. Suppression des accents (décomposition Unicode)
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    # 3. Remplacement de la ponctuation par des espaces
    text = re.sub(r"[^\w\s]", " ", text)

    # 4. Suppression des espaces superflus (multiples -> un seul)
    text = re.sub(r"\s+", " ", text).strip()

    print("query cleanning", query)

    return text


class RedisCacheService:
    @staticmethod
    def _hash_query(query: str) -> str:
        """Génère une clé de hachage unique pour le cache exact."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @classmethod
    def _exact_cache_key(cls, query: str, tenant_id: str, user_role: str) -> str:
        """Isole chaque entrée exacte par entreprise et rôle utilisateur."""
        scope = f"{tenant_id.strip().lower()}:{user_role.strip().lower()}"
        scope_hash = hashlib.sha256(scope.encode("utf-8")).hexdigest()
        return f"cache:exact:{scope_hash}:{cls._hash_query(query)}"

    @classmethod
    async def get_cached_answer(
        cls, query: str, tenant_id: str = "default", user_role: str = "all"
    ) -> dict | None:
        """
        Vérifie d'abord le cache exact (via hash), puis sémantique si configuré.
        Retourne la réponse si trouvée, sinon None.
        """
        print(f"[Cache] Vérification du cache pour la requête : '{query}'")
        try:
            client = await get_redis_client()
            query_clean = normalize_query(query)
            # 1. Vérification du cache exact
            exact_key = cls._exact_cache_key(query_clean, tenant_id, user_role)
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
    async def set_cached_answer(
        cls,
        query: str,
        answer_data: dict,
        ttl: int = 3600,
        tenant_id: str = "default",
        user_role: str = "all",
    ) -> bool:
        """
        Enregistre une réponse dans le cache Redis avec un TTL (par défaut 1 heure).
        """
        print("[Cache] Mise en cache de la réponse...")
        try:
            client = await get_redis_client()
            query_clean = normalize_query(query)
            exact_key = cls._exact_cache_key(query_clean, tenant_id, user_role)

            # Stockage de la réponse sérialisée en JSON
            await client.setex(exact_key, ttl, json.dumps(answer_data))
            print(f"[Cache] Réponse mise en cache pour : '{query}' (TTL: {ttl}s)")
            log.info(
                f"[Cache] Réponse mise en cache pour : '{query},{tenant_id},{user_role}' (TTL: {ttl}s)"
            )
            return True
        except Exception as e:
            log.error(f"[Cache Error] Erreur lors de l'écriture en cache : {e}")
            return False
