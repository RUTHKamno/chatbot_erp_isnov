"""Tenant-scoped storage for Cube YAML schemas in LanceDB and Cube's repository."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

import lancedb
import requests

from app.ingestion.settings import get_cfg

logger = logging.getLogger(__name__)
TABLE_NAME = "tenant_semantic_models"
_index_lock = threading.Lock()


def _tenant_key(tenant_id: str) -> str:
    from app.ingestion.pipeline import normalize_tenant_id

    return normalize_tenant_id(tenant_id)[:100]


def _validate_yaml_files(files: dict[str, str]) -> dict[str, str]:
    if not files:
        raise ValueError("Chargez au moins un fichier YAML de modèle Cube.")

    validated: dict[str, str] = {}
    for filename, content in files.items():
        if (
            not filename
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
            or Path(filename).suffix.lower() not in {".yaml", ".yml"}
        ):
            raise ValueError(f"Nom de fichier YAML invalide : {filename!r}.")
        if not content.strip():
            raise ValueError(f"Le fichier {filename} est vide.")
        if len(content.encode("utf-8")) > 1_000_000:
            raise ValueError(f"Le fichier {filename} dépasse la limite de 1 Mo.")
        key = filename.casefold()
        if key in validated:
            raise ValueError(f"Le fichier {filename} est fourni plusieurs fois.")
        validated[key] = content
    return {Path(name).name: content for name, content in files.items()}


def _embed(text: str) -> list[float]:
    cfg = get_cfg()
    response = requests.post(
        f"{cfg.ollama_base_url}/api/embeddings",
        json={"model": cfg.embed_model, "prompt": text, "keep_alive": "30m"},
        timeout=120,
    )
    response.raise_for_status()
    embedding = response.json().get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("Ollama a retourné un embedding YAML vide.")
    return [float(value) for value in embedding]


def tenant_cube_schema_directory(tenant_id: str) -> Path:
    tenant_id = _tenant_key(tenant_id)
    if not re.fullmatch(r"[a-z0-9_-]{1,100}", tenant_id):
        raise ValueError("Identifiant tenant non valide pour le dépôt Cube.")
    return Path(__file__).resolve().parent / tenant_id / "model" / "cubes"


def replace_tenant_cube_schemas(
    tenant_id: str, files: dict[str, str]
) -> dict[str, str]:
    """Replace only the tenant's YAML schemas and return the previous contents."""
    validated = _validate_yaml_files(files)
    target_dir = tenant_cube_schema_directory(tenant_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    previous = {
        path.name: path.read_text(encoding="utf-8")
        for path in target_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    }

    staged_dir = Path(
        tempfile.mkdtemp(prefix=".cube-yaml-", dir=str(target_dir.parent))
    )
    try:
        for filename, content in validated.items():
            (staged_dir / filename).write_text(content, encoding="utf-8")

        for old_file in target_dir.iterdir():
            if old_file.is_file() and old_file.suffix.lower() in {".yaml", ".yml"}:
                old_file.unlink()
        for staged_file in staged_dir.iterdir():
            os.replace(staged_file, target_dir / staged_file.name)
    except OSError:
        restore_tenant_cube_schemas(tenant_id, previous)
        raise
    finally:
        for staged_file in staged_dir.iterdir():
            staged_file.unlink(missing_ok=True)
        staged_dir.rmdir()
    return previous


def restore_tenant_cube_schemas(tenant_id: str, files: dict[str, str]) -> None:
    target_dir = tenant_cube_schema_directory(tenant_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in target_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            path.unlink()
    for filename, content in files.items():
        (target_dir / filename).write_text(content, encoding="utf-8")


def index_tenant_cube_models(tenant_id: str, files: dict[str, str]) -> int:
    """Embed and replace only one tenant's YAML models in the semantic LanceDB table."""
    tenant_id = _tenant_key(tenant_id)
    if not tenant_id:
        raise ValueError("Identifiant tenant invalide.")
    validated = _validate_yaml_files(files)
    rows: list[dict[str, Any]] = []
    for filename, content in validated.items():
        model_name = Path(filename).stem
        semantic_text = f"Modèle Cube {model_name}.\n{content}"
        rows.append(
            {
                "id": f"{tenant_id}_{model_name}",
                "vector": _embed(semantic_text),
                "tenant_id": tenant_id,
                "table_name": model_name,
                "yaml_content": content,
                "description": f"Modèle sémantique Cube {model_name}",
            }
        )

    db = lancedb.connect(get_cfg().lancedb_uri)
    with _index_lock:
        if TABLE_NAME in db.list_tables().tables:
            table = db.open_table(TABLE_NAME)
            escaped_tenant = tenant_id.replace("'", "''")
            previous_rows = (
                table.search().where(f"tenant_id = '{escaped_tenant}'").to_list()
            )
            table.delete(f"tenant_id = '{escaped_tenant}'")
            try:
                table.add(rows)
            except Exception:
                logger.exception("Échec d’indexation LanceDB pour %s", tenant_id)
                if previous_rows:
                    table.add(previous_rows)
                raise
        else:
            db.create_table(TABLE_NAME, data=rows)
    return len(rows)


def get_tenant_cube_models(tenant_id: str) -> list[dict[str, str]]:
    tenant_id = _tenant_key(tenant_id)
    if not tenant_id:
        return []
    db = lancedb.connect(get_cfg().lancedb_uri)
    if TABLE_NAME not in db.list_tables().tables:
        return []
    escaped_tenant = tenant_id.replace("'", "''")
    rows = (
        db.open_table(TABLE_NAME)
        .search()
        .where(f"tenant_id = '{escaped_tenant}'")
        .select(["table_name", "yaml_content"])
        .to_list()
    )
    return [
        {"table_name": row["table_name"], "yaml_content": row["yaml_content"]}
        for row in rows
    ]
