# Fichier : app/agents/reformulator.py
import ast
import logging
from typing import Any

from nemoguardrails import LLMRails, RailsConfig

from app.graph.state import Intent, SMAState
from app.guardrails.actions import (
    check_redis_cache,
    reformulate_query,
    check_input_toxicity,
    detect_sql_injection,
)

log = logging.getLogger("agent.reformulator")

config = RailsConfig.from_path("app/guardrails")
rails_app = LLMRails(config)

rails_app.register_action(reformulate_query)
rails_app.register_action(check_redis_cache)
rails_app.register_action(check_input_toxicity)
rails_app.register_action(detect_sql_injection)


def _response_to_text(response: Any) -> str:
    """Extract assistant text from the response shapes returned by NeMo."""
    content = getattr(response, "response", response)

    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = content.get("content", "")
        return content if isinstance(content, str) else str(content or "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                item_content = item.get("content", "")
            else:
                item_content = getattr(item, "content", item)
            if item_content is not None:
                parts.append(
                    item_content
                    if isinstance(item_content, str)
                    else str(item_content)
                )
        return "\n".join(part for part in parts if part)

    return str(content) if content is not None else ""


async def reformulate_and_security_node(state: SMAState) -> dict:
    """
    Nœud LangGraph unifié enrichi et corrigé.
    """
    raw_query = state.get("raw_query", "")
    messages = state.get("messages", [])
    jwt_claims = state.get("jwt_claims", {})
    tenant_id = jwt_claims.get("tenant_id", "default")

    print(f"\n--- [DEBUG REFORMULATOR START] ---")
    print(
        f"[Reformulator] Requête brute reçue : '{raw_query}' pour le tenant '{tenant_id}'"
    )

    nemo_messages = []
    for m in messages:
        if hasattr(m, "type") and hasattr(m, "content"):
            role = "user" if m.type == "human" else "assistant"
            content = m.content
        elif isinstance(m, (tuple, list)) and len(m) >= 2:
            role = "user" if m[0] == "user" else "assistant"
            content = m[1]
        else:
            role = "user"
            content = str(m)
        nemo_messages.append({"role": role, "content": content})

    nemo_context = {"_history": nemo_messages[:-1], "user_message": raw_query}

    try:
        # Exécution globale de NeMo (Input + Dialog)
        response = await rails_app.generate_async(
            messages=nemo_messages, options={"context": nemo_context}
        )

        # ---------------------------------------------------------------------
        # 🩹 CORRECTION : Normalisation sécurisée de la réponse de NeMo
        # ---------------------------------------------------------------------
        bot_response = _response_to_text(response)

        # On s'assure que c'est une chaîne de caractères propre pour éviter le crash
        bot_response_str = bot_response.lower().strip()

        # Extraction depuis le contexte NeMo
        detected_intent = nemo_context.get("detected_intent")
        reformulated = nemo_context.get("reformulated_query")

        # ---------------------------------------------------------------------
        # 🩹 CORRECTION RADICALE : Extraction des listes du contexte NeMo
        # ---------------------------------------------------------------------
        # Si NeMo a stocké l'intention sous forme de liste (historique de variable)
        if isinstance(detected_intent, list):
            detected_intent = detected_intent[-1] if len(detected_intent) > 0 else ""

        # Si la requête reformulée est aussi emballée dans une liste
        if isinstance(reformulated, list):
            reformulated = reformulated[-1] if len(reformulated) > 0 else ""

        # Conversion et nettoyage stricts en chaînes de caractères
        intent_str = str(detected_intent).lower().strip() if detected_intent else ""
        reformulated_str = str(reformulated).strip() if reformulated else raw_query
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # 🔒 SÉCURISATION : Détection multi-critères des menaces
        # ---------------------------------------------------------------------
        is_blocked_by_text = (
            "malheureusement, je ne peux pas traiter" in bot_response_str
            or "alerte de sécurité" in bot_response_str
            or "restons courtois" in bot_response_str
        )

        if intent_str == "security_threat" or is_blocked_by_text:
            print(
                "[Reformulator] 🛑 ALERTE SÉCURITÉ ACTIVE -> Route forcée : 'blocked'"
            )
            print("--- [DEBUG REFORMULATOR END] ---\n")
            return {
                "raw_query": raw_query,
                "reformulated_query": raw_query,
                "is_safe": False,
                "intent": "security_threat",
                "route": "blocked",
                "answer": bot_response
                or "Requête bloquée par les filtres de sécurité.",
            }

        # 2. Gestion des intentions conversationnelles directes
        if (
            intent_str == "greeting"
            or intent_str == "gratitude"
            or intent_str == "off_topic"
        ):
            print(
                f"[Reformulator] 💬 Route conversationnelle interceptée : {intent_str}"
            )
            print(f"--- [DEBUG REFORMULATOR END] ---\n")
            return {
                "reformulated_query": raw_query,
                "is_safe": True,
                "intent": intent_str,
                "route": "blocked",  # On bloque le RAG et on s'arrête (END)
                "answer": bot_response,
            }

        # Si le cache Redis a répondu positivement
        cache_hit_data = nemo_context.get("cache_result")
        if cache_hit_data and cache_hit_data.get("hit"):
            print(f"[Reformulator] ⚡ Cache HIT détecté.")
            print(f"--- [DEBUG REFORMULATOR END] ---\n")
            return {
                "reformulated_query": reformulated,
                "is_safe": True,
                "intent": "cache_hit",
                "route": "blocked",
                "answer": cache_hit_data.get("answer"),
            }

        # 3. Intentions Métier
        if not intent_str:
            intent_str = "user_guide"

        route_mapping = {
            "user_guide": "guide_rag",
            "error_ticket": "guide_rag",
            "bi_analyst": "bi",
        }
        target_route = route_mapping.get(intent_str, "guide_rag")

        print(
            f"[Reformulator] ✅ Intention métier validée : '{intent_str}' -> Route : '{target_route}'"
        )
        print(f"--- [DEBUG REFORMULATOR END] ---\n")

        return {
            "reformulated_query": reformulated,
            "is_safe": True,
            "intent": intent_str,
            "route": target_route,
            "answer": None,
        }

    except Exception as e:
        print(f"[Reformulator Critical Error] {e}")
        log.error(f"[Reformulator Error] {e}")
        print(f"--- [DEBUG REFORMULATOR END] ---\n")
        return {
            "raw_query": raw_query,
            "reformulated_query": raw_query,
            "is_safe": False,
            "intent": "unknown",
            "route": "blocked",
            "answer": "Une erreur technique est survenue lors de l'analyse.",
        }
