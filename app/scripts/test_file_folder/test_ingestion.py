from pathlib import Path
from app.ingestion.loaders import LOADERS
from app.ingestion.pipeline import IngestionService, format_structured_text

def test_pipeline():
    file_path = Path("app/scripts/test_file_folder/test_guide.md")
    source_type = "text"  # Ou "tabular" pour le CSV, "pdf" pour le PDF

    print("--- 1. TEST DU LOADER ---")
    loader = LOADERS[source_type]
    extracted_doc = loader(file_path)

    for i, block in enumerate(extracted_doc.blocks):
        print(f"\n[Bloc {i+1}]")
        print(f"Texte : {block.text}")
        print(f"Métadonnées extraites : {block.meta}")

    print("\n--- 2. TEST DE LA MISE EN FORME SÉMANTIQUE ---")
    block = extracted_doc.blocks[0]
    roles = block.meta.get("roles", ["all"])
    formatted = format_structured_text(
        role=", ".join(roles),
        ecran=block.meta.get("ecran", ""),
        module=block.meta.get("module", ""),
        instructions=block.text
    )
    print("Texte enrichi pour Embedding :")
    print(formatted)

if __name__ == "__main__":
    test_pipeline()