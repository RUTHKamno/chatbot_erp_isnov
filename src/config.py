import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SOURCE_PDFS_DIR = DATA_DIR / "source_pdfs"
LANCEDB_PATH = str(BASE_DIR / "lancedb_data")

# --- NOUVEAU : Chemins pour la couche sémantique Cube ---
# Dossier source où Cube crache ses YAML (depuis ton conteneur ou ton montage VeraCrypt)
CUBE_SOURCE_MODEL_DIR = Path(os.environ.get(
    "CUBE_SOURCE_DIR", 
    "/media/veracrypt1/dev_env_ruthka/AI-BI/AI/PROJECTS/app/core/isnov/model"
))

# Dossier local sécurisé pour les YAML validés/modifiés par l'utilisateur
SEMANTIC_CURATED_DIR = DATA_DIR / "semantic_curated"
SEMANTIC_CURATED_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "qwen2.5:7b")
LLM_MODEL_CODER_NAME = os.environ.get("LLM_MODEL_CODER_NAME", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LANCEDB_TABLE_NAME = "business_knowledge"