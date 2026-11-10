import os
import shutil
import yaml
from pathlib import Path
from config import CUBE_SOURCE_MODEL_DIR, SEMANTIC_CURATED_DIR

def sync_cube_models_to_curated():
    """
    Synchronise les fichiers YAML de Cube vers le dossier curé local.
    Si un fichier existe déjà localement, on NE l'écrase PAS pour préserver 
    les modifications et synonymes de l'utilisateur.
    """
    if not CUBE_SOURCE_MODEL_DIR.exists():
        print(f"[ATTENTION] Le dossier source Cube '{CUBE_SOURCE_MODEL_DIR}' est introuvable. Vérifie le chemin.")
        return False

    synced_count = 0
    for file_path in CUBE_SOURCE_MODEL_DIR.glob("**/*.yml"):
        relative_path = file_path.relative_to(CUBE_SOURCE_MODEL_DIR)
        destination_path = SEMANTIC_CURATED_DIR / relative_path
        
        # Créer les sous-dossiers si nécessaire
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        # Si le fichier n'existe pas encore dans le dossier curé, on le copie
        if not destination_path.exists():
            shutil.copy2(file_path, destination_path)
            synced_count += 1

    print(f"[SUCCÈS] Synchronisation sémantique : {synced_count} nouveaux fichiers importés.")
    return True

def get_all_curated_models() -> list[str]:
    """Renvoie la liste de tous les fichiers YAML disponibles dans le dossier curé."""
    if not SEMANTIC_CURATED_DIR.exists():
        return []
    return [str(p.relative_to(SEMANTIC_CURATED_DIR)) for p in SEMANTIC_CURATED_DIR.glob("**/*.yml")]

def load_yaml_model(relative_filename: str) -> dict:
    """Charge un fichier YAML sémantique sous forme de dictionnaire Python."""
    file_path = SEMANTIC_CURATED_DIR / relative_filename
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_yaml_model(relative_filename: str, data: dict):
    """Sauvegarde les modifications (synonymes, calculs, etc.) dans le fichier YAML curé."""
    file_path = SEMANTIC_CURATED_DIR / relative_filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"[SUCCÈS] Modèle sémantique mis à jour et sauvegardé : {relative_filename}")