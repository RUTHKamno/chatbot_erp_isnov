# Fichier : app/graph/workflow.py
import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from app.agents.bi_analyst_node import bi_analyst_node
from app.agents.erp_guide import guide_rag_node
from app.agents.evaluator import evaluator_judge_node

# Import des nœuds agents
from app.agents.reformulator import reformulate_and_security_node
from app.agents.voice_handler import voice_processing_node
from app.core.cache_service import RedisCacheService

# Import de l'État et des Intentions
from app.graph.state import SMAState

log = logging.getLogger(__name__)


# --- 1. Condition de routage dynamique depuis le Reformulateur ---
def route_after_reformulation(
    state: SMAState,
) -> Literal["guide_rag", "bi_node", "__end__"]:
    """
    Aiguille dynamiquement le flux.
    """
    # Correction de la valeur par défaut : 'guide_rag' cohérent avec le nœud reformulator
    route = state.get("route", "guide_rag")

    if route in ["blocked", "greeting", "off_topic", "cache"]:
        print(f"[Router] 🛑 Court-circuit activé vers END (Raison: {route})")
        return END

    elif route == "guide_rag":
        print("[Router] 🧭 Direction le nœud RAG : guide_rag")
        return "guide_rag"

    elif route == "bi":
        print("[Router] 📊 Direction le nœud BI analyste")
        return "bi_analyst"

    return "guide_rag"


# --- 2. Condition de routage de la boucle de rétroaction (Évaluateur) ---
def should_continue(
    state: SMAState,
) -> Literal["guide_rag", "cache_answer", "__end__"]:
    """
    Logique de routage de la boucle de rétroaction :
    - Si la réponse passe l'évaluation, le workflow se termine.
    - Si l'évaluation échoue mais qu'il reste des essais (retry_count < 3), réoriente vers 'guide_rag'.
    """
    passed = state.get("evaluation_passed", False)
    retries = state.get("retry_count", 0)

    if passed:
        return "cache_answer"

    if retries >= 3:
        return END

    return "guide_rag"


async def cache_validated_answer_node(state: SMAState) -> dict:
    """Cache une réponse ERP Guide uniquement après validation du graphe."""
    query = state.get("reformulated_query") or state.get("raw_query", "")
    answer = state.get("answer", "")
    jwt_claims = state.get("jwt_claims", {}) or {}

    if (
        state.get("evaluation_passed") is not True
        or not state.get("retrieved_docs")
        or not query.strip()
        or not answer.strip()
    ):
        return {}

    non_cacheable_prefixes = (
        "Erreur",
        "Désolé, aucune documentation",
        "Je ne dispose pas de la documentation",
        "Je ne trouve pas de documentation",
        "Désolé, la réponse générée n'a pas pu être validée",
    )
    if answer.lstrip().startswith(non_cacheable_prefixes):
        return {}

    stored = await RedisCacheService.set_cached_answer(
        query,
        {"answer": answer},
        tenant_id=jwt_claims.get("tenant_id", "default"),
        user_role=jwt_claims.get("role", "all"),
    )
    if stored:
        log.info("Réponse ERP Guide validée mise en cache.")
    else:
        log.warning("Échec de mise en cache de la réponse ERP Guide validée.")
    return {}


# ---------------- après l'éxécution du noeud BI, si le process est allée jusqu'a la fin, on doit stocker le contenu dans le cache
# ---------------- pour accélrer le processus de réponses par l'agent bi_analyst_tool --------------------
async def cache_bi_query_node(state: SMAState) -> dict:
    """Cache uniquement la requête SQL (Cube query) après une analyse BI réussie."""
    query = state.get("reformulated_query") or state.get("raw_query", "")
    sql_query = state.get("sql_query", "")
    jwt_claims = state.get("jwt_claims", {}) or {}

    if not query.strip() or not sql_query.strip():
        return {}

    print(
        "in cache store",
        query,
        jwt_claims.get("tenant_id"),
        jwt_claims.get("role"),
        sql_query,
    )

    stored = await RedisCacheService.set_cached_answer(
        query,
        {"sql_query": sql_query},  # On stocke le SQL et non la réponse textuelle
        tenant_id=jwt_claims.get("tenant_id", "default"),
        user_role=jwt_claims.get("role", "all"),
    )
    if stored:
        log.info("Requête SQL (Cube query) BI mise en cache avec succès.")
    else:
        log.warning("Échec de mise en cache de la requête SQL BI.")
    return {}


# --- 3. Construction et Assemblage du Graphe ---
workflow = StateGraph(SMAState)

# Ajout des nœuds
workflow.add_node("voice_handler", voice_processing_node)
workflow.add_node("reformulator", reformulate_and_security_node)
workflow.add_node("guide_rag", guide_rag_node)
workflow.add_node("evaluator", evaluator_judge_node)
workflow.add_node("cache_answer", cache_validated_answer_node)
workflow.add_node("bi_analyst", bi_analyst_node)
workflow.add_node("cache_bi_node", cache_bi_query_node)  # noeud du cache BI

# Le point d'entrée du système est désormais le Nœud Reformulateur & Sécurité
# workflow.set_entry_point("reformulator")
workflow.set_entry_point("voice_handler")

# Transition conditionnelle depuis le reformulateur vers le reste du graphe
workflow.add_edge("voice_handler", "reformulator")
workflow.add_conditional_edges(
    "reformulator",
    route_after_reformulation,
    {
        "guide_rag": "guide_rag",
        "bi_analyst": "bi_analyst",
        END: END,
    },
)

# Transition du Guide RAG vers l'Évaluateur
workflow.add_edge("guide_rag", "evaluator")

# Arête conditionnelle depuis l'évaluateur (Validation vs Boucle de régénération)
workflow.add_conditional_edges(
    "evaluator",
    should_continue,
    {
        "guide_rag": "guide_rag",
        "cache_answer": "cache_answer",
        END: END,
    },
)

# Ne pas retarder la réponse au client : l'écriture s'exécute en fin de parcours.
workflow.add_edge("cache_answer", END)

# --------- après bi_analyst, on appelle le cache , après on ferme/boucle le graph ------------
workflow.add_edge("bi_analyst", "cache_bi_node")
workflow.add_edge("cache_bi_node", END)

# Compilation du Graphe
sma_graph = workflow.compile()
