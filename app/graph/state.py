"""LangGraph shared memory (agent state)."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

""" 
class Intent(str, Enum):
    GREETING = "greeting"
    OFF_TOPIC = "off_topic"
    SECURITY_THREAT = "security_threat"
    CACHE_HIT = "cache_hit"
    USER_GUIDE = "user_guide"
    BI_ANALYST = "bi_analyst"
    ARTIFACT_REQUEST = "artifact_request"
    UNKNOWN = "unknown"
"""


class Intent(str, Enum):
    GREETING = "greeting"
    OFF_TOPIC = "off_topic"
    SECURITY_THREAT = "security_threat"
    CACHE_HIT = "cache_hit"
    USER_GUIDE = "user_guide"
    ERROR_TICKET = "error_ticket"  # Ajout pour la gestion des erreurs/tickets
    BI_ANALYST = "bi_analyst"
    UNAUTHORIZED = "unauthorized"  # Ajout pour le blocage RBAC
    UNKNOWN = "unknown"


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    REFORMULATOR = "reformulator"
    SQL_GENERATOR = "sql_generator"
    EVALUATOR = "evaluator"
    ARTIFACTS = "artifacts"


class TraceStep(TypedDict, total=False):
    agent: str
    action: str
    detail: str
    latency_ms: float
    ok: bool


class SMAState(TypedDict, total=False):
    """Shared memory passed between LangGraph nodes."""

    # Request context
    messages: Annotated[list, add_messages]
    user_id: str
    session_id: str
    raw_query: str
    jwt_claims: dict[str, Any]
    current_page: Any
    system_prompt: str

    # Routing
    intent: str
    route: Literal[
        "greeting",
        "off_topic",
        "blocked",
        "cache",
        "guide",
        "bi",
        "artifacts",
        "end",
    ]

    # Reformulator / security
    reformulated_query: str
    is_safe: bool
    security_flags: list[str]

    # Cache
    cache_hit: bool
    cached_response: dict[str, Any]

    # RAG / guide
    retrieved_docs: list[dict[str, Any]]
    confidence_score: float  # Score de similarité/confiance (0.0 à 1.0)
    low_confidence_count: int  # Compteur d'échecs consécutifs sous 70%
    suggestions: list[dict[str, str]]  # Boutons de suggestion renvoyés au front

    # SQL / Wren
    sql_query: str
    sql_result: dict[str, Any]
    wren_metadata: dict[str, Any]

    # Evaluator
    evaluation_passed: bool
    evaluation_feedback: str
    retry_count: int

    # Artifacts
    artifact_type: str
    artifact_paths: list[str]

    # Final answer
    answer: str
    sources: list[str]
    debug_trace: list[TraceStep]
    error: str
