import lancedb
import requests
from config import LANCEDB_PATH, LANCEDB_TABLE_NAME, EMBEDDING_MODEL_NAME

def get_ollama_embedding(text: str) -> list[float]:
    """Interroge le conteneur Ollama local pour obtenir l'embedding."""
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": "nomic-embed-text:latest",  # Utilise le modèle présent dans Docker
        "prompt": text
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        raise Exception(f"Erreur Ollama API ({response.status_code}): {response.text}")

def retrieve_business_context(cleaned_query: str, domain_intent: str, limit: int = 3) -> list[str]:
    db = lancedb.connect(LANCEDB_PATH)
    
    if LANCEDB_TABLE_NAME not in db.table_names():
        print(f"[ATTENTION] La table '{LANCEDB_TABLE_NAME}' n'existe pas dans LanceDB.")
        return []

    table = db.open_table(LANCEDB_TABLE_NAME)

    print(f"[INFO] Vectorisation de la question via Ollama ({EMBEDDING_MODEL_NAME})...")
    # Génération de l'embedding via Ollama au lieu de SentenceTransformer
    query_vector = get_ollama_embedding(cleaned_query)

    # Recherche avec filtre de domaine
    query_builder = table.search(query_vector).limit(limit)
    if domain_intent and domain_intent != "general":
        query_builder = query_builder.where(f"domain = '{domain_intent}'", prefilter=True)

    results = query_builder.to_df()

    if results.empty:
        results = table.search(query_vector).limit(limit).to_df()

    if results.empty:
        return []

    return results["text"].tolist()