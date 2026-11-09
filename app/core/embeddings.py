"""Ollama embedding helper for LanceDB vector search."""

from __future__ import annotations

import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Ollama `/api/embeddings`."""
    settings = get_settings()
    vectors: list[list[float]] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for text in texts:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/embeddings",
                json={"model": settings.ollama_embed_model, "prompt": text},
            )
            resp.raise_for_status()
            embedding = resp.json().get("embedding")
            if not embedding:
                raise RuntimeError("Ollama returned an empty embedding")
            vectors.append([float(x) for x in embedding])
    return vectors


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]
