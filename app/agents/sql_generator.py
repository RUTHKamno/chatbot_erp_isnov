"""SQL generator agent + Wren AI Engine adapter."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.agents.reformulator import assistant_disclaimer
from app.config import get_settings
from app.graph.state import SMAState

logger = logging.getLogger(__name__)


class WrenAIAdapter:
    """HTTP adapter toward Wren AI service (text-to-SQL / ask)."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def ask(self, question: str, project_id: str | None = None) -> dict[str, Any]:
        """
        Call Wren AI ask endpoint.

        Endpoint shape may vary by Wren version; this adapter isolates that
        so the LangGraph node stays stable.
        """
        base = self.settings.wren_ai_service_url.rstrip("/")
        payload = {
            "query": question,
            "project_id": project_id or self.settings.wren_project_id or None,
        }
        # Strip nulls
        payload = {k: v for k, v in payload.items() if v is not None}

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Primary: Wren AI service ask API
            for path in ("/v1/asks", "/v1/ask", "/ask"):
                try:
                    resp = await client.post(f"{base}{path}", json=payload)
                    if resp.status_code < 400:
                        data = resp.json()
                        return {"ok": True, "endpoint": path, "data": data}
                except httpx.HTTPError as exc:
                    logger.debug("Wren path %s failed: %s", path, exc)

        # Fallback stub for local scaffolding when Wren is unreachable
        return {
            "ok": False,
            "endpoint": None,
            "data": {
                "sql": (
                    "-- Wren AI unavailable — stub SQL\n"
                    "SELECT SUM(amount) AS chiffre_affaires\n"
                    "FROM sales\n"
                    "WHERE period = CURRENT_DATE;"
                ),
                "summary": (
                    "Le service Wren AI n'est pas joignable. "
                    "Voici une esquisse SQL illustrative uniquement."
                ),
                "stub": True,
            },
        }

    async def run_sql(self, sql: str) -> dict[str, Any]:
        """Optional: push SQL to Wren engine / ibis for execution."""
        base = self.settings.wren_engine_url.rstrip("/")
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(f"{base}/v1/sql", json={"sql": sql})
                if resp.status_code < 400:
                    return {"ok": True, "result": resp.json()}
                return {"ok": False, "error": resp.text}
            except httpx.HTTPError as exc:
                return {"ok": False, "error": str(exc)}


async def sql_generator_node(state: SMAState) -> dict[str, Any]:
    """Generate SQL via Wren AI and prepare analytical answer draft."""
    started = time.perf_counter()
    settings = get_settings()
    question = state.get("reformulated_query") or state.get("raw_query") or ""
    trace = list(state.get("debug_trace") or [])

    adapter = WrenAIAdapter()
    wren = await adapter.ask(question)
    data = wren.get("data") or {}

    sql = data.get("sql") or data.get("query") or ""
    if not sql and isinstance(data.get("response"), dict):
        sql = data["response"].get("sql", "")

    summary = (
        data.get("summary")
        or data.get("answer")
        or data.get("content")
        or "Analyse BI générée via Wren AI."
    )

    sql_result: dict[str, Any] = {"wren": data}
    if sql and not data.get("stub"):
        executed = await adapter.run_sql(sql)
        sql_result["execution"] = executed

    answer = (
        f"{summary}\n\n"
        f"**Requête SQL proposée :**\n```sql\n{sql or 'N/A'}\n```\n\n"
        f"_{assistant_disclaimer()}_"
    )

    latency = (time.perf_counter() - started) * 1000
    if settings.enable_debug_trace:
        trace.append(
            {
                "agent": "sql_generator",
                "action": "wren_ask",
                "detail": f"ok={wren.get('ok')} endpoint={wren.get('endpoint')} sql_len={len(sql)}",
                "latency_ms": round(latency, 2),
                "ok": bool(wren.get("ok")) or bool(sql),
            }
        )

    return {
        "sql_query": sql,
        "sql_result": sql_result,
        "wren_metadata": {
            "ok": wren.get("ok"),
            "endpoint": wren.get("endpoint"),
            "stub": bool(data.get("stub")),
        },
        "answer": answer,
        "sources": ["wren_ai", "dwh"],
        "debug_trace": trace,
        "route": "bi",
    }
