import csv
import re
import sys
import requests
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path



import pdfplumber
 
# Add project root to sys.path to enable importing `app` when run from subdirectories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

 
from app.config import get_settings
from app.ingestion.store import vector_store
from app.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------- lecture du PDF
FOOTER_RE = re.compile(r"^Guide utilisateur ERP - Page \d+$")
MODULE_HEADING_RE = re.compile(r"^\d+\. Module \S")
STEP_RE = re.compile(r"^\d+\. ")
 
 
def parse_guide_pdf(path: str) -> List[Dict[str, Any]]:
    lines: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for line in (page.extract_text() or "").splitlines():
                line = line.strip()
                if line and not FOOTER_RE.match(line) and not MODULE_HEADING_RE.match(line):
                    lines.append(line)
 
    guides: List[Dict[str, Any]] = []
    cur: Dict[str, Any] | None = None
    paras: List[str] = []
 
    def close():
        if cur is not None:
            cur["instructions"] = "\n".join(paras).strip()
            guides.append(cur)
 
    for line in lines:
        if line.startswith("Écran : "):
            close()
            cur, paras = {"ecran": line[8:].strip(), "module": "", "roles": [], "url": ""}, []
        elif cur is None:
            continue  # page de garde, présentation, tableau des rôles
        elif line.startswith("Module : "):
            cur["module"] = line[9:].strip()
        elif line.startswith("Rôles autorisés : "):
            cur["roles"] = [r.strip() for r in line[18:].split(",")]
        elif line.startswith("URL : "):
            cur["url"] = line[6:].strip()
        elif not paras or STEP_RE.match(line) or line.startswith("Procédure"):
            paras.append(line)            # nouveau paragraphe / nouvelle étape
        else:
            paras[-1] += " " + line       # ligne de continuation (retour à la ligne du PDF)
    close()
    return guides
 
 
 
# ---------------------------------------------------------------- chargement
def load_csv_guides(path: str) -> List[Dict[str, Any]]:
    """CSV attendu : colonnes role (ou roles), ecran, module, instructions, url.
    Plusieurs rôles possibles dans une cellule, séparés par ; , ou |"""
    guides = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw_roles = row.get("roles") or row.get("role") or ""
            guides.append({
                "roles": [r.strip() for r in re.split(r"[;,|]", raw_roles) if r.strip()],
                "ecran": row["ecran"].strip(),
                "module": row["module"].strip(),
                "instructions": row["instructions"].strip(),
                "url": (row.get("url") or "").strip(),
            })
    return guides
 
 
def load_guides(path: str) -> List[Dict[str, Any]]:
    if path.lower().endswith(".pdf"):
        return parse_guide_pdf(path)
    if path.lower().endswith(".csv"):
        return load_csv_guides(path)
    raise ValueError(f"Format non supporté : {path} (attendu : .pdf ou .csv)")
 
 
# ---------------------------------------------------------------- embeddings
def format_structured_text(role: str, ecran: str, module: str, instructions: str) -> str:
    """Mise en forme sémantique du texte pour maximiser la qualité de l'embedding."""
    return (
        f"L'utilisateur disposant du rôle [{role}] est autorisé à effectuer des opérations "
        f"sur l'écran [{ecran}] du module [{module}]. "
        f"Pour réaliser ces actions, la procédure à suivre est la suivante : {instructions}"
    )
 
 
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
 
 
# ---------------------------------------------------------------- indexation
def process_and_seed_guides(raw_guides: List[Dict[str, Any]]):
    """Découpe les guides en chunks, génère les embeddings et insère dans LanceDB.
 
    Le filtre RBAC de vector_store utilise une seule valeur `role_autorise` par ligne :
    un écran accessible à plusieurs rôles produit donc une ligne par rôle
    (l'embedding est calculé une seule fois et réutilisé).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
 
    data_to_insert = []
    chunk_id = 1
 
    for guide in raw_guides:
        roles = guide["roles"]
        # 1. Découpage des instructions seules, puis mise en forme de CHAQUE chunk
        #    (ainsi chaque chunk garde l'écran, le module et le rôle en contexte)
        raw_chunks = text_splitter.split_text(guide["instructions"])
        label_roles = ", ".join(roles)
 
        for raw_chunk in raw_chunks:
            chunk = format_structured_text(
                role=label_roles,
                ecran=guide["ecran"],
                module=guide["module"],
                instructions=raw_chunk,
            )
            vector = get_embedding(chunk)  # une seule fois par chunk
 
            # 2. Une ligne par rôle autorisé ('all' reste 'all')
            for role in roles:
                data_to_insert.append({
                    "id": f"doc_{chunk_id}",
                    "vector": vector,
                    "text": chunk,
                    "role_autorise": role,
                    "ecran": guide["ecran"],
                    "module": guide["module"],
                    "url": guide.get("url", ""),
                })
                chunk_id += 1
 
    vector_store.init_table(data_to_insert)
    print(f"✅ {len(data_to_insert)} lignes indexées dans LanceDB "
          f"({len(raw_guides)} écrans lus).")
 
 
if __name__ == "__main__":
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        default_data_pdf = PROJECT_ROOT / "data" / "guide_erp_documentation.pdf"
        source = str(default_data_pdf) if default_data_pdf.exists() else "guide_erp_documentation.pdf"

    guides = load_guides(source)
    print(f"{len(guides)} écrans chargés depuis {source}")
    process_and_seed_guides(guides)
 