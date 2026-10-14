"""Application configuration loaded from .env via os.getenv (no duplicated defaults)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Project root directory: parent of app/ (e.g. /media/veracrypt1/AI-BI/AI/PROJECTS)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env once at import time (search from project root, then fallback to CWD)
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()


def _resolve_path(raw_path: str) -> str:
    path = Path(raw_path)
    if not path.is_absolute():
        return str((PROJECT_ROOT / path).resolve())
    return str(path)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_bool(key: str, default: str = "false") -> bool:
    return _env(key, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: str = "0") -> int:
    return int(_env(key, default))


def _env_float(key: str, default: str = "0.0") -> float:
    return float(_env(key, default))


class Settings:
    """Runtime settings — values come from the environment / .env file."""

    # API
    app_name: str = _env("APP_NAME", "Chatbot ERP Multi-Agent")
    app_env: str = _env("APP_ENV", "development")
    app_debug: bool = _env_bool("APP_DEBUG", "true")
    api_host: str = _env("API_HOST", "0.0.0.0")
    api_port: int = _env_int("API_PORT", "8000")
    api_prefix: str = _env("API_PREFIX", "/api/v1")

    # JWT
    jwt_secret_key: str = _env("JWT_SECRET_KEY", "change-me")
    jwt_algorithm: str = _env("JWT_ALGORITHM", "HS256")
    jwt_expire_minutes: int = _env_int("JWT_EXPIRE_MINUTES", "60")

    # Redis
    redis_host: str = _env("REDIS_HOST", "localhost")
    redis_port: int = _env_int("REDIS_PORT", "6379")
    redis_db: int = _env_int("REDIS_DB", "0")
    redis_password: str = _env("REDIS_PASSWORD", "")
    cache_ttl_seconds: int = _env_int("CACHE_TTL_SECONDS", "3600")
    cache_similarity_threshold: float = _env_float("CACHE_SIMILARITY_THRESHOLD", "0.92")

    # PostgreSQL
    postgres_host: str = _env("POSTGRES_HOST", "localhost")
    postgres_port: int = _env_int("POSTGRES_PORT", "5433")
    postgres_user: str = _env("POSTGRES_USER", "erp_user")
    postgres_password: str = _env("POSTGRES_PASSWORD", "erp_secure_password")
    postgres_db: str = _env("POSTGRES_DB", "erp_dwh")
    database_url: str = _env(
        "DATABASE_URL",
        "postgresql+asyncpg://erp_user:erp_secure_password@localhost:5433/erp_dwh",
    )

    # LanceDB
    lancedb_uri: str = _resolve_path(_env("LANCEDB_URI", "./data/lancedb"))
    lancedb_table: str = _env("LANCEDB_TABLE", "erp_knowledge")

    # Ollama
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_llm_model: str = _env("OLLAMA_LLM_MODEL", "llama3.1:8b")
    ollama_llm_guide_erp_model: str = _env("OLLAMA_LLM_GUIDE_ERP_MODEL", "qwen2.5:7b")
    ollama_embed_model: str = _env("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_guard_model: str = _env("OLLAMA_GUARD_MODEL", "llama-guard3:8b")

    # Wren AI
    wren_ai_service_url: str = _env("WREN_AI_SERVICE_URL", "http://localhost:5555")
    wren_ui_url: str = _env("WREN_UI_URL", "http://localhost:3000")
    wren_engine_url: str = _env("WREN_ENGINE_URL", "http://localhost:8080")
    wren_project_id: str = _env("WREN_PROJECT_ID", "")

    # Guardrails
    nemo_guardrails_config_path: str = _resolve_path(
        _env("NEMO_GUARDRAILS_CONFIG_PATH", "./app/guards/nemo_config")
    )
    enable_llama_guard: bool = _env_bool("ENABLE_LLAMA_GUARD", "true")
    enable_sql_injection_filter: bool = _env_bool("ENABLE_SQL_INJECTION_FILTER", "true")
    evaluator_strict_mode: bool = _env_bool("EVALUATOR_STRICT_MODE", "true")

    # Artifacts
    artifacts_output_dir: str = _resolve_path(
        _env("ARTIFACTS_OUTPUT_DIR", "./data/artifacts")
    )
    artifacts_max_pptx_slides: int = _env_int("ARTIFACTS_MAX_PPTX_SLIDES", "20")

    # Observability
    log_level: str = _env("LOG_LEVEL", "INFO")
    enable_debug_trace: bool = _env_bool("ENABLE_DEBUG_TRACE", "true")

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def artifacts_path(self) -> Path:
        path = Path(self.artifacts_output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
