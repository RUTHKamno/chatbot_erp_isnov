# Fichier : app/graph/workflow.py
from typing import Literal

from langgraph.graph import END, StateGraph

from app.agents.erp_guide import guide_rag_node
from app.agents.evaluator import evaluator_judge_node

# Import des nœuds agents
from app.agents.reformulator import reformulate_and_security_node

# Import de l'État et des Intentions
from app.graph.state import SMAState


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
        return END

    return "guide_rag"


# --- 2. Condition de routage de la boucle de rétroaction (Évaluateur) ---
def should_continue(state: SMAState) -> Literal["guide_rag", "__end__"]:
    """
    Logique de routage de la boucle de rétroaction :
    - Si la réponse passe l'évaluation, le workflow se termine.
    - Si l'évaluation échoue mais qu'il reste des essais (retry_count < 3), réoriente vers 'guide_rag'.
    """
    passed = state.get("evaluation_passed", False)
    retries = state.get("retry_count", 0)

    if passed or retries >= 3:
        return END

    return "guide_rag"


# --- 3. Construction et Assemblage du Graphe ---
workflow = StateGraph(SMAState)

# Ajout des nœuds
workflow.add_node("reformulator", reformulate_and_security_node)
workflow.add_node("guide_rag", guide_rag_node)
workflow.add_node("evaluator", evaluator_judge_node)

# Le point d'entrée du système est désormais le Nœud Reformulateur & Sécurité
workflow.set_entry_point("reformulator")

# Transition conditionnelle depuis le reformulateur vers le reste du graphe
workflow.add_conditional_edges(
    "reformulator",
    route_after_reformulation,
    {
        "guide_rag": "guide_rag",
        "bi_node": END,  # Mappé proprement pour éviter toute erreur de clé LangGraph
        END: END,
    },
)

# Transition du Guide RAG vers l'Évaluateur
workflow.add_edge("guide_rag", "evaluator")

# Arête conditionnelle depuis l'évaluateur (Validation vs Boucle de régénération)
workflow.add_conditional_edges(
    "evaluator", should_continue, {"guide_rag": "guide_rag", END: END}
)

# Compilation du Graphe
sma_graph = workflow.compile()
