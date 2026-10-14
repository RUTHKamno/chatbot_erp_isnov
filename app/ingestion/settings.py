"""Configuration de l'ingestion. Lit app.config une seule fois, de façon paresseuse
(les tests peuvent donc appeler configure() sans jamais importer app.config)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class IngestConfig:
    data_dir: Path                    # uploads + registre SQLite
    lancedb_uri: str
    ollama_base_url: str
    ollama_model: str
    embed_model: str
    max_upload_mb: int = 25
    chunk_size: int = 900
    chunk_overlap: int = 150
    csv_rows_per_chunk: int = 25
    csv_max_rows: int = 2000          # au-delà : tronqué + avertissement
    embed_retries: int = 2
    admin_roles: Tuple[str, ...] = ("admin",)   # peuvent voir/supprimer les documents de leur entreprise


_cfg: Optional[IngestConfig] = None


def get_cfg() -> IngestConfig:
    global _cfg
    if _cfg is None:
        from app.config import get_settings, PROJECT_ROOT  # import tardif
        s = get_settings()
        _cfg = IngestConfig(
            data_dir=Path(PROJECT_ROOT) / "data",
            lancedb_uri=s.lancedb_uri,
            ollama_base_url=s.ollama_base_url,
            embed_model=s.ollama_embed_model,
            ollama_model = s.ollama_llm_model
        )
    return _cfg


def configure(cfg: IngestConfig) -> None:
    global _cfg
    _cfg = cfg
