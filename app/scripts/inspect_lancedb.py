import argparse
import sys
from pathlib import Path

import lancedb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings

settings = get_settings()

TABLE_NAME = "erp_enterprises"
META_FIELDS = [
    ("id", "ID"),
    ("tenant_id", "Entreprise (tenant)"),
    ("visibility", "Visibilité"),
    ("module", "Module"),
    ("ecran", "Écran"),
    ("url", "URL"),
    ("source_name", "Fichier source"),
    ("doc_id", "Doc ID"),
    ("page", "Page"),
    ("chunk_index", "Index du chunk"),
    ("ingested_at", "Ingéré le"),
]


def inspect_database(tenant: str | None = None, text_chars: int = 200) -> None:
    """Affiche les chunks de la table erp_enterprises avec leurs métadonnées."""
    print(f"🔍 Connexion à la base LanceDB : {settings.lancedb_uri}")
    db = lancedb.connect(str(settings.lancedb_uri))

    if TABLE_NAME not in db.table_names():
        print(f"⚠️ La table '{TABLE_NAME}' n'existe pas encore (aucune ingestion réussie ?).")
        return

    table = db.open_table(TABLE_NAME)
    df = table.to_pandas()
    print(f"📊 Total dans '{TABLE_NAME}' : {len(df)} chunks")

    if tenant:
        df = df[df["tenant_id"] == tenant.strip().lower()]
        print(f"🏢 Filtre tenant = '{tenant}' → {len(df)} chunks")

    if df.empty:
        print("Aucun chunk à afficher.")
        return

    # Résumé par entreprise
    print("\n--- 🏢 RÉPARTITION PAR ENTREPRISE ---")
    for name, count in df.groupby("tenant_id").size().items():
        print(f"  • {name} : {count} chunks")

    print("\n--- 📄 CHUNKS ET MÉTADONNÉES ---")
    for _, row in df.sort_values(["tenant_id", "source_name", "chunk_index"]).iterrows():
        print(f"\n[Chunk {row.get('chunk_index', '?')}]")
        for key, label in META_FIELDS:
            print(f"  • {label:<20}: {row.get(key, 'N/A')}")

        roles = [r for r in str(row.get("allowed_roles", "")).split("|") if r]
        print(f"  • {'Rôles autorisés':<20}: {', '.join(roles) if roles else 'N/A'}")

        text = str(row.get("text", "")).replace("\n", " ")
        print(f"  • {'Extrait du texte':<20}: {text[:text_chars]}{'...' if len(text) > text_chars else ''}")

        vec = row.get("vector")
        if vec is not None:
            print(f"  • {'Dim. vectorielle':<20}: {len(vec)} dimensions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspecter la table erp_enterprises")
    parser.add_argument("--tenant", help="Filtrer sur une entreprise (ex: dolibarr)")
    parser.add_argument("--chars", type=int, default=200, help="Longueur de l'extrait de texte")
    args = parser.parse_args()
    inspect_database(tenant=args.tenant, text_chars=args.chars)