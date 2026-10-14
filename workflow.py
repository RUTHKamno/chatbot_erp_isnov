"""LangGraph workflow — multi-agent orchestration for the ERP chatbot."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.agents.artifacts import artifacts_node, guide_node
from app.agents.evaluator import evaluator_node
from app.agents.orchestrator import orchestrator_node
from app.agents.reformulator import reformulator_node
from app.agents.sql_generator import sql_generator_node
from app.core.redis_client import get_redis
from app.graph.state import SMAState


async def greeting_node(state: SMAState) -> dict[str, Any]:
    return {
        "answer": (
            "Bonjour ! Je suis votre assistant ERP et BI. "
            "Je peux vous guider dans l'application ou vous aider à comprendre vos indicateurs. "
            "Comment puis-je vous aider ?"
        ),
        "sources": ["orchestrator"],
    }


async def off_topic_node(state: SMAState) -> dict[str, Any]:
    return {
        "answer": (
            "Cette question semble hors du périmètre ERP / BI. "
            "Reformulez autour de vos tâches dans l'application ou de vos indicateurs métier."
        ),
        "sources": ["orchestrator"],
    }


async def blocked_node(state: SMAState) -> dict[str, Any]:
    return {
        "answer": state.get("answer")
        or "Requête bloquée pour des raisons de sécurité.",
        "sources": ["security"],
    }


async def cache_node(state: SMAState) -> dict[str, Any]:
    cached = state.get("cached_response") or {}
    return {
        "answer": cached.get("answer") or state.get("answer") or "",
        "sources": cached.get("sources") or ["cache"],
        "artifact_paths": cached.get("artifact_paths") or [],
    }


async def persist_cache_node(state: SMAState) -> dict[str, Any]:
    """Store successful answers in Redis for later interception."""
    if state.get("cache_hit"):
        return {}
    answer = state.get("answer")
    if not answer:
        return {}
    user_id = state.get("user_id") or "anonymous"
    query = state.get("raw_query") or ""
    try:
        redis = get_redis()
        await redis.set_cached_answer(
            user_id,
            query,
            {
                "answer": answer,
                "sources": state.get("sources") or [],
                "artifact_paths": state.get("artifact_paths") or [],
                "intent": state.get("intent"),
            },
        )
    except Exception:  # noqa: BLE001
        pass
    return {}


def _after_orchestrator(
    state: SMAState,
) -> Literal["reformulator", "cache", "greeting", "off_topic", "end"]:
    route = state.get("route")
    if route == "cache":
        return "cache"
    if route == "greeting":
        return "greeting"
    if route == "off_topic":
        return "off_topic"
    if route in {"guide", "bi", "artifacts"}:
        return "reformulator"
    return "end"


def _after_reformulator(
    state: SMAState,
) -> Literal["guide", "sql_generator", "artifacts", "blocked", "end"]:
    if not state.get("is_safe", True) or state.get("route") == "blocked":
        return "blocked"
    route = state.get("route")
    if route == "guide":
        return "guide"
    if route == "bi":
        return "sql_generator"
    if route == "artifacts":
        return "artifacts"
    return "end"


def _after_evaluator(
    state: SMAState,
) -> Literal["sql_generator", "persist_cache", "end"]:
    if (
        not state.get("evaluation_passed", True)
        and (state.get("retry_count") or 0) < 2
        and state.get("route") == "bi"
    ):
        return "sql_generator"
    if state.get("evaluation_passed", True) and state.get("answer"):
        return "persist_cache"
    return "end"


def build_sma_graph() -> Any:
    """Compile the LangGraph StateGraph for the SMA chatbot."""
    graph = StateGraph(SMAState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("reformulator", reformulator_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("off_topic", off_topic_node)
    graph.add_node("blocked", blocked_node)
    graph.add_node("cache", cache_node)
    graph.add_node("guide", guide_node)
    graph.add_node("sql_generator", sql_generator_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("artifacts", artifacts_node)
    graph.add_node("persist_cache", persist_cache_node)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        _after_orchestrator,
        {
            "reformulator": "reformulator",
            "cache": "cache",
            "greeting": "greeting",
            "off_topic": "off_topic",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "reformulator",
        _after_reformulator,
        {
            "guide": "guide",
            "sql_generator": "sql_generator",
            "artifacts": "artifacts",
            "blocked": "blocked",
            "end": END,
        },
    )
    graph.add_edge("sql_generator", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        _after_evaluator,
        {
            "sql_generator": "sql_generator",
            "persist_cache": "persist_cache",
            "end": END,
        },
    )
    graph.add_edge("guide", "evaluator")
    graph.add_edge("artifacts", "persist_cache")
    graph.add_edge("greeting", END)
    graph.add_edge("off_topic", END)
    graph.add_edge("blocked", END)
    graph.add_edge("cache", END)
    graph.add_edge("persist_cache", END)

    return graph.compile()


# Lazily compiled singleton
_compiled = None


def get_sma_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_sma_graph()
    return _compiled
