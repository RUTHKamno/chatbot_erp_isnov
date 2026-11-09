# Fichier : app/agents/reformulator.py
import ast
import logging
import re
from typing import Any
import unicodedata

from nemoguardrails import LLMRails, RailsConfig

from app.graph.state import Intent, SMAState
from app.guardrails.actions import (
    block_graph_of_toxic,
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
rails_app.register_action(block_graph_of_toxic)


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
                    item_content if isinstance(item_content, str) else str(item_content)
                )
        return "\n".join(part for part in parts if part)

    return str(content) if content is not None else ""


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)  # ponctuation et apostrophes -> espace
    return re.sub(r"\s+", " ", text).strip()


async def reformulate_and_security_node(state: SMAState) -> dict:
    """
    Nœud LangGraph unifié : Exécute NeMo Guardrails en s'appuyant sur prompts.yml,
    nettoie les variables du contexte, applique une heuristique anti-jailbreak,
    puis renvoie le dictionnaire d'état et la route appropriée.
    """
    raw_query = state.get("raw_query", "")
    messages = state.get("messages", [])
    jwt_claims = state.get("jwt_claims", {}) or {}
    tenant_id = jwt_claims.get("tenant_id", "default")
    user_role = jwt_claims.get("role", "all")

    print(f"\n--- [DEBUG REFORMULATOR START] ---")
    print(
        f"[Reformulator] Requête brute reçue : '{raw_query}' pour le tenant '{tenant_id}'"
    )

    # =====================================================================
    # ⚡ INTERCEPTION DIRECTE DES CAS LIMITES (0 NeMo, 0 LLM, Route immédiate)
    # =====================================================================
    query_stripped = str(raw_query).strip()
    query_normalized = _normalize(
        query_stripped
    )  # Utilise votre fonction de normalisation existante

    # --- L01 : Message totalement vide ---
    if not query_stripped:
        print("[Reformulator] ⚠️ Cas limite L01 (Vide) détecté -> Court-circuit.")
        return {
            "raw_query": raw_query,
            "reformulated_query": raw_query,
            "is_safe": True,
            "intent": "limit_case",
            "route": "blocked",
            "answer": "Bonjour ! Votre message semble vide. Comment puis-je vous aider aujourd'hui avec votre logiciel ERP ?",
        }

    # --- L02 : Ponctuation isolée ou point d'interrogation ---
    if query_normalized in ["", "?", "!", "??", "???"]:
        print(
            "[Reformulator] ⚠️ Cas limite L02 (Ponctuation) détecté -> Court-circuit."
        )
        return {
            "raw_query": raw_query,
            "reformulated_query": raw_query,
            "is_safe": True,
            "intent": "limit_case",
            "route": "blocked",
            "answer": "Besoin d'aide ? N'hésitez pas à me poser une question complète, par exemple : 'Comment créer une facture ?' ou 'Comment annuler une commande ?'.",
        }

    # --- L03 : Confirmation simple isolée ---
    if query_normalized in ["ok", "d accord", "parfait", "super"]:
        print(
            "[Reformulator] ⚠️ Cas limite L03 (Confirmation) détecté -> Court-circuit."
        )
        return {
            "raw_query": raw_query,
            "reformulated_query": raw_query,
            "is_safe": True,
            "intent": "limit_case",
            "route": "blocked",
            "answer": "Très bien ! Je reste à votre entière disposition si vous avez la moindre question sur l'utilisation de vos modules ERP.",
        }

    # --- L05 : Uniquement des Émojis ---
    if not re.search(r"[a-z0-9]", query_normalized) and any(
        c for c in query_stripped if ord(c) > 0x1F600
    ):
        print("[Reformulator] ⚠️ Cas limite L05 (Émojis) détecté -> Court-circuit.")
        return {
            "raw_query": raw_query,
            "reformulated_query": raw_query,
            "is_safe": True,
            "intent": "limit_case",
            "route": "blocked",
            "answer": "Bonjour ! Comment puis-je vous assister dans vos opérations de gestion ou l'utilisation de l'ERP aujourd'hui ? 😀",
        }

    # --- L04 : Charabia évident / Spam clavier ---
    is_spam_keyboard = any(
        spam in query_normalized for spam in ["azertyuiop", "qsdfghjklm", "wxcvbn"]
    )
    is_no_vowels = len(query_normalized) > 8 and not re.search(
        r"[aeiouy]", query_normalized
    )

    if is_spam_keyboard or is_no_vowels:
        print("[Reformulator] ⚠️ Cas limite L04 (Charabia) détecté -> Court-circuit.")
        return {
            "raw_query": raw_query,
            "reformulated_query": raw_query,
            "is_safe": True,
            "intent": "limit_case",
            "route": "blocked",
            "answer": "Je n'ai pas compris votre demande. Pourriez-vous reformuler votre question en utilisant des termes liés à vos tâches de gestion sur l'ERP ?",
        }

    # Préparation de l'historique des messages pour NeMo Guardrails
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

    nemo_context = {
        "_history": nemo_messages[:-1],
        "user_message": raw_query,
        "tenant_id": tenant_id,
        "user_role": user_role,
    }

    try:
        # Exécution de la chaîne globale NeMo (Input Rails -> prompts.yml intent classification -> Dialog Rails)

        response = await rails_app.generate_async(
            messages=nemo_messages,
            options={
                "context": nemo_context,
                "output_vars": [
                    "detected_intent",
                    "reformulated_query",
                    "cache_result",
                ],
            },
        )

        # Extraction sécurisée de la réponse textuelle du Bot
        bot_response = _response_to_text(response)
        bot_response_str = bot_response.lower().strip()

        # NeMo renvoie les variables Colang dans output_data (Colang 1.0).
        output_data = getattr(response, "output_data", None)
        if isinstance(output_data, dict):
            nemo_context.update(output_data)
        elif output_data is not None:
            log.warning(
                "Format inattendu pour les variables NeMo retournées: %s",
                type(output_data).__name__,
            )

        # Récupération des variables du contexte retourné par le flow.
        detected_intent = nemo_context.get("detected_intent")
        reformulated = nemo_context.get("reformulated_query")

        print(f"[Reformulator] Intention détectée : '{detected_intent}'")
        print(f"[Reformulator] Requête reformulée : '{reformulated}'")
        print(f"[Reformulator] Contexte NeMo : {nemo_context}")

        # ---------------------------------------------------------------------
        # 🩹 NETTOYAGE STRICT : Désenveloppement des listes du contexte NeMo
        # ---------------------------------------------------------------------
        if isinstance(detected_intent, list):
            detected_intent = detected_intent[-1] if len(detected_intent) > 0 else ""

        if isinstance(reformulated, list):
            reformulated = reformulated[-1] if len(reformulated) > 0 else ""

        intent_str = str(detected_intent).lower().strip() if detected_intent else ""
        reformulated_str = str(reformulated).strip() if reformulated else raw_query

        # ---------------------------------------------------------------------
        # 🔒 ÉVALUATION DE LA ROUTE SÉCURITÉ ACTIVE
        # ---------------------------------------------------------------------
        is_blocked_by_text = (
            "malheureusement, je ne peux pas traiter" in bot_response_str
            or "alerte de sécurité" in bot_response_str
            or "restons courtois" in bot_response_str
        )

        if intent_str == "security_threat" or is_blocked_by_text:
            print(
                "[Reformulator] 🛑 ALERTE SÉCURITÉ ACTIVE (NeMo) -> Route forcée : 'blocked'"
            )
            print("--- [DEBUG REFORMULATOR END] ---\n")
            return {
                "raw_query": raw_query,
                "reformulated_query": raw_query,
                "is_safe": False,
                "intent": "security_threat",
                "route": "blocked",
                "answer": bot_response
                or "Requête bloquée par les politiques de modération de sécurité.",
            }

        # ---------------------------------------------------------------------
        # 💬 ÉVALUATION DES ROUTES CONVERSATIONNELLES SIMPLES (Politesse & Hors-Sujet)
        # ---------------------------------------------------------------------
        is_blocked_by_text_greeting = (
            "bonjour ! je suis votre assistant" in bot_response_str
            or "je suis spécialisé exclusivement dans l'assistance" in bot_response_str
            or "comment puis-je vous aider aujourd'hui ?" in bot_response_str
            or "je ne peux pas répondre aux sujets hors de ce périmètre."
            in bot_response_str
        )
        if (
            intent_str in ["greeting", "gratitude", "off_topic"]
            or is_blocked_by_text_greeting
        ):
            print(
                f"[Reformulator] 💬 Route conversationnelle interceptée : '{intent_str}' -> Route : 'blocked'"
            )
            print("--- [DEBUG REFORMULATOR END] ---\n")
            return {
                "raw_query": raw_query,
                "reformulated_query": raw_query,
                "is_safe": True,
                "intent": intent_str,
                "route": "blocked",  # Bloque l'accès au nœud RAG et redirige vers END
                "answer": bot_response
                or "Interaction prise en charge par l'assistant d'accueil.",
            }

        # ---------------------------------------------------------------------
        # ⚡ ÉVALUATION DU CACHE REDIS (S'il a déjà répondu positivement)
        # ---------------------------------------------------------------------
        cache_hit_data = nemo_context.get("cache_result")
        if isinstance(cache_hit_data, list):
            cache_hit_data = cache_hit_data[-1] if cache_hit_data else None
        if isinstance(cache_hit_data, dict) and cache_hit_data.get("hit"):
            print(f"[Reformulator] ⚡ Cache HIT détecté.")
            print("--- [DEBUG REFORMULATOR END] ---\n")
            return {
                "raw_query": raw_query,
                "reformulated_query": reformulated_str,
                "is_safe": True,
                "intent": "cache_hit",
                "route": "blocked",
                "answer": cache_hit_data.get("answer"),
            }

        # ---------------------------------------------------------------------
        # 🧭 AIGUILLAGE NOMINAL DES INTENTIONS MÉTIER (RAG ERP & BI)
        # ---------------------------------------------------------------------
        if not intent_str:
            intent_str = "user_guide"  # Intention par défaut si aucune menace/politesse n'est détectée

        route_mapping = {
            "user_guide": "guide_rag",
            "error_ticket": "guide_rag",
            "bi_analyst": "bi",
        }
        target_route = route_mapping.get(intent_str, "guide_rag")

        print(
            f"[Reformulator] ✅ Intention métier validée : '{intent_str}' -> Route : '{target_route}'"
        )
        print("--- [DEBUG REFORMULATOR END] ---\n")

        return {
            "raw_query": raw_query,
            "reformulated_query": reformulated_str,
            "is_safe": True,
            "intent": intent_str,
            "route": target_route,
            "answer": None,  # Transmis au nœud de génération RAG pour traitement
        }

    except Exception as e:
        print(f"[Reformulator Critical Error] {e}")
        log.error(f"[Reformulator Error] {e}")
        print("--- [DEBUG REFORMULATOR END] ---\n")
        return {
            "raw_query": raw_query,
            "reformulated_query": raw_query,
            "is_safe": False,
            "intent": "unknown",
            "route": "blocked",
            "answer": "Une erreur technique interne est survenue lors de l'analyse de votre requête.",
        }
