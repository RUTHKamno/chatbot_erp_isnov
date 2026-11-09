# Fichier : app/scripts/ingest_yaml_schemas.py
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import lancedb
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print("[project_root]", PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings

settings = get_settings()


def get_embedding(text: str) -> List[float]:
    """Génère l'embedding via le modèle nomic-embed-text sur Ollama."""
    response = requests.post(
        f"{settings.ollama_base_url}/api/embeddings",
        json={"model": settings.ollama_embed_model, "prompt": text},
        timeout=60,
    )
    if response.status_code == 200:
        return response.json()["embedding"]
    raise Exception(f"Erreur d'embedding Ollama: {response.text}")


def load_cube_yamls(models_dir: str) -> List[Dict[str, Any]]:
    """Charge chaque fichier YAML du dossier des cubes en entier (pas de chunking)."""
    yaml_files = glob.glob(os.path.join(models_dir, "*.yaml")) + glob.glob(
        os.path.join(models_dir, "*.yml")
    )
    print(yaml_files, "----------------------------------")
    yamls_data = []

    for file_path in yaml_files:
        filename = os.path.basename(file_path)
        print("in load cube yamls filename", filename)
        cube_name = os.path.splitext(filename)[0]
        print("in load cube yamls cube_name", filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            yamls_data.append(
                {"cube_name": cube_name, "content": content, "filename": filename}
            )
    return yamls_data


def index_tenant_cube_models(tenant_id: str, models_dir: str):
    """
    Ingère les fichiers YAML complets pour un tenant donné dans LanceDB,
    sans découpage par chunks, garantissant l'intégrité du schéma sémantique.
    """
    raw_yamls = load_cube_yamls(models_dir)
    db_path = PROJECT_ROOT / "data" / "core" / ".lancedb"
    db = lancedb.connect(str(db_path))

    data_to_insert = []

    for item in raw_yamls:
        # Texte sémantique enrichi pour l'embedding du fichier complet
        semantic_text = f"Cube sémantique pour l'entité {item['cube_name']}.\nContenu YAML:\n{item['content']}"

        vector = get_embedding(semantic_text)

        data_to_insert.append(
            {
                "id": f"{tenant_id}_{item['cube_name']}",
                "vector": vector,
                "tenant_id": tenant_id,
                "table_name": item["cube_name"],
                "yaml_content": item["content"],
                "description": f"Modèle sémantique complet pour {item['cube_name']}",
            }
        )

    table_name = "tenant_semantic_models"
    if table_name in db.table_names():
        table = db.open_table(table_name)
        # Nettoyage optionnel des anciennes entrées du tenant avant ré-indexation
        # table.delete(f"tenant_id = '{tenant_id}'")
        table.add(data_to_insert)
    else:
        db.create_table(table_name, data=data_to_insert)

    print(
        f"✅ {len(data_to_insert)} modèles YAML indexés en bloc (sans chunking) pour le tenant '{tenant_id}'."
    )


# if __name__ == "__main__":
#    print("indexing yaml files")
#    index_tenant_cube_models()
