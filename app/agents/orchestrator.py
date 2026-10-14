"""Orchestrator agent — intent routing and cache interception."""

from __future__ import annotations

import logging
import time
from typing import Any

from app.config import get_settings
from app.core.redis_client import get_redis
from app.graph.state import Intent, SMAState

logger = logging.getLogger(__name__)

_GREETING_HINTS = ("bonjour", "hello", "salut", "bonsoir", "hey", "coucou")
_GUIDE_HINTS = (
    "où cliquer",
    "ou cliquer",
    "comment faire",
    "où aller",
    "ou aller",
    "bouton",
    "menu",
    "naviguer",
    "profil",
    "écran",
    "ecran",
    "page",
)
_BI_HINTS = (
    "chiffre d'affaire",
    "chiffre d'affaires",
    "ca ",
    "vente",
    "ventes",
    "marge",
    "kpi",
    "rapport",
    "analyse",
    "pourquoi",
    "évolution",
    "evolution",
    "stock",
    "client",
    "commande",
)
_ARTIFACT_HINTS = ("powerpoint", "pptx", "pdf", "visualisation", "graphique", "export", "jasper")
_OFFTOPIC_HINTS = ("météo", "meteo", "foot", "recette", "blague")


def _classify_intent(query: str) -> Intent:
    q = query.lower().strip()
    if any(h in q for h in _GREETING_HINTS) and len(q.split()) <= 4:
        return Intent.GREETING
    if any(h in q for h in _OFFTOPIC_HINTS):
        return Intent.OFF_TOPIC
    if any(h in q for h in _ARTIFACT_HINTS):
        return Intent.ARTIFACT_REQUEST
    if any(h in q for h in _GUIDE_HINTS):
        return Intent.USER_GUIDE
    if any(h in q for h in _BI_HINTS):
        return Intent.BI_ANALYST
    # Default to BI for analytical phrasing, otherwise guide
    if "?" in q and any(w in q for w in ("combien", "quel", "quelle", "total")):
        return Intent.BI_ANALYST
    return Intent.USER_GUIDE


def _intent_to_route(intent: Intent) -> str:
    mapping = {
        Intent.GREETING: "greeting",
        Intent.OFF_TOPIC: "off_topic",
        Intent.SECURITY_THREAT: "blocked",
        Intent.CACHE_HIT: "cache",
        Intent.USER_GUIDE: "guide",
        Intent.BI_ANALYST: "bi",
        Intent.ARTIFACT_REQUEST: "artifacts",
        Intent.UNKNOWN: "guide",
    }
    return mapping[intent]


async def orchestrator_node(state: SMAState) -> dict[str, Any]:
    """Entry node: check cache, classify intent, set route."""
    started = time.perf_counter()
    settings = get_settings()
    query = state.get("raw_query") or ""
    user_id = state.get("user_id") or "anonymous"
    trace = list(state.get("debug_trace") or [])

    # 1) Exact semantic-cache lookup (Redis)
    cache_hit = False
    cached_response: dict[str, Any] = {}
    try:
        redis = get_redis()
        cached = await redis.get_cached_answer(user_id, query)
        if cached:
            cache_hit = True
            cached_response = cached
            intent = Intent.CACHE_HIT
        else:
            intent = _classify_intent(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache lookup failed: %s", exc)
        intent = _classify_intent(query)

    route = _intent_to_route(intent)
    latency = (time.perf_counter() - started) * 1000

    if settings.enable_debug_trace:
        trace.append(
            {
                "agent": "orchestrator",
                "action": "route",
                "detail": f"intent={intent.value} route={route} cache_hit={cache_hit}",
                "latency_ms": round(latency, 2),
                "ok": True,
            }
        )

    updates: dict[str, Any] = {
        "intent": intent.value,
        "route": route,
        "cache_hit": cache_hit,
        "debug_trace": trace,
        "retry_count": state.get("retry_count") or 0,
    }
    if cache_hit:
        updates["cached_response"] = cached_response
        updates["answer"] = cached_response.get("answer", "")
        updates["sources"] = cached_response.get("sources", ["cache"])
    return updates
