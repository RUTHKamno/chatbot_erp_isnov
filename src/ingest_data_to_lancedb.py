import os
from pathlib import Path
import lancedb
import requests
from pypdf import PdfReader
from config import LANCEDB_PATH, LANCEDB_TABLE_NAME, EMBEDDING_MODEL_NAME, SOURCE_PDFS_DIR

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def get_ollama_embedding(text: str) -> list:
    """Appelle le conteneur Ollama pour générer l'embedding."""
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": "nomic-embed-text:latest",
        "prompt": text
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        raise Exception(f"Erreur Ollama API ({response.status_code}): {response.text}")

def ingest_pdf_to_lancedb(pdf_filename: str, domain_key: str):
    pdf_path = SOURCE_PDFS_DIR / pdf_filename
    if not pdf_path.exists():
        print(f"[ERREUR] Fichier introuvable : {pdf_path}")
        return

    print(f"[INFO] Extraction du texte depuis {pdf_filename}...")
    text = extract_text_from_pdf(str(pdf_path))
    
    # Découpage simple par paragraphes du guide métier
    chunks = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
    print(f"[INFO] {len(chunks)} blocs de texte extraits.")

    print(f"[INFO] Génération des embeddings via le conteneur Ollama (nomic-embed-text)...")
    
    data = []
    for i, chunk in enumerate(chunks):
        formatted_text = f"Thématique [{domain_key.upper()}]: {chunk}"
        embedding = get_ollama_embedding(formatted_text)
        
        data.append(
            {
                "id": f"{domain_key}_{i}",
                "domain": domain_key,
                "text": formatted_text,
                "vector": embedding
            }
        )
        print(f" -> Bloc {i+1}/{len(chunks)} vectorisé.")

    db = lancedb.connect(LANCEDB_PATH)
    if LANCEDB_TABLE_NAME in db.table_names():
        table = db.open_table(LANCEDB_TABLE_NAME)
        table.add(data)
    else:
        db.create_table(LANCEDB_TABLE_NAME, data=data)
        
    print(f"[SUCCÈS] {len(data)} blocs indexés sous la clé domain='{domain_key}' via Ollama.")

if __name__ == "__main__":
    pdf_name = "regles_stock.pdf"
    domain = "stock"
    ingest_pdf_to_lancedb(pdf_name, domain)