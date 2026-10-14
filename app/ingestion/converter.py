import io
import logging
import json
from pathlib import Path
from typing import Generator
import requests
import pdfplumber
from pptx import Presentation
import pandas as pd

from app.ingestion import security
from app.ingestion.settings import get_cfg

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ 1. EXTRACTION BRUTE (STREAMING / PAGE PAR PAGE)
def stream_raw_text(
    file_path: Path, filename: str, pages_per_chunk: int = 5
) -> Generator[str, None, None]:
    """
    Lit le fichier et génère des morceaux de texte brut (chunks)
    afin de pouvoir traiter les documents très volumineux (ex: 1000+ pages).
    """
    file_path = Path(file_path)
    source_type = security.detect_source_type(file_path, filename)

    if source_type == "pdf":
        with pdfplumber.open(str(file_path)) as pdf:
            buffer = []
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    buffer.append(f"--- [Page {i}] ---\n{text}")
                if len(buffer) >= pages_per_chunk:
                    yield "\n\n".join(buffer)
                    buffer = []
            if buffer:
                yield "\n\n".join(buffer)

    elif source_type == "pptx":
        prs = Presentation(str(file_path))
        buffer = []
        for i, slide in enumerate(prs.slides, 1):
            slide_texts = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                    if shape.text_frame.text.strip():
                        slide_texts.append(shape.text_frame.text.strip())
            if slide_texts:
                buffer.append(f"--- [Diapositive {i}] ---\n" + "\n".join(slide_texts))
            if len(buffer) >= pages_per_chunk:
                yield "\n\n".join(buffer)
                buffer = []
        if buffer:
            yield "\n\n".join(buffer)

    elif source_type == "csv":
        # Pour les gros CSV, lecture par morceaux (chunks) de lignes avec pandas
        for chunk_df in pd.read_csv(file_path, chunksize=100):
            lines = []
            for _, row in chunk_df.iterrows():
                line = " | ".join(
                    f"{c}: {v}"
                    for c, v in row.items()
                    if pd.notna(v) and str(v).strip()
                )
                if line:
                    lines.append(line)
            if lines:
                yield "\n".join(lines)

    elif source_type == "text":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for line in f:
                lines.append(line)
                if len(lines) >= 200:  # ~200 lignes par bloc
                    yield "".join(lines)
                    lines = []
            if lines:
                yield "".join(lines)


# ------------------------------------------------------------------ 2. ANALYSE ET REFORMATAGE PAR LE LLM
# app/ingestion/converter.py

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "modules": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Main software module names only (e.g. Tiers-Société, Mailing, Imports)",
        },
        "roles": {
            "type": "array",
            "items": {"type": "string"},
            "description": "All job titles, actors, user roles, or functions mentioned or target of the feature",
        },
        "screens": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific screens, menu paths, tabs or action names",
        },
    },
    "required": ["modules", "roles", "screens"],
}


def analyze_page_with_llm(page_text: str) -> dict:
    cfg = get_cfg()

    prompt = f"""You are a precise software documentation metadata extractor.

Task: Extract modules, user roles/job functions, and screens/features from the text.

CRITICAL INSTRUCTION FOR ROLES:
Look for ANY job title, user type, persona, or permission mentioned or affected by the text (e.g., "administrateur", "PDG", "ouvrier", "vendeur", "trésorier", "service contentieux", "contact facturation").
Do NOT output ["all"] if any of these specific job titles or functions are present in the text.

Text to analyze:
<<<
{page_text}
>>>"""

    try:
        response = requests.post(
            f"{cfg.ollama_base_url}/api/generate",
            json={
                "model": cfg.ollama_model,  # Ex: qwen2.5:7b ou deepseek-r1:8b (avec nettoyage)
                "prompt": prompt,
                "format": JSON_SCHEMA,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 4096,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        raw_res = response.json().get("response", "{}")

        data = json.loads(raw_res)

        modules = data.get("modules", [])
        roles = data.get("roles", [])
        screens = data.get("screens", [])

        return {
            "module": ", ".join(modules) if modules else "Non spécifié",
            "roles": ", ".join(roles) if roles else "all",
            "screen": ", ".join(screens) if screens else "Non spécifié",
        }
    except Exception as e:
        log.error(f"Erreur d'analyse LLM : {e}")
        return {"module": "Non spécifié", "roles": "all", "screen": "Non spécifié"}


def extract_pages_or_chunks(file_path: Path) -> Generator[tuple[int, str], None, None]:
    """
    Générateur polymorphe qui extrait le texte page par page (PDF)
    ou section par section / bloc (TXT, CSV, HTML scrapé, PPTX).
    """
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        with pdfplumber.open(str(file_path)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    yield idx, page_text.strip()

    elif ext in (".txt", ".md", ".log"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()

        # Si le fichier contient des délimiteurs de pages web / sections scrapées
        sections = [s.strip() for s in content.split("--- [Page ") if s.strip()]
        if len(sections) > 1:
            for idx, section in enumerate(sections, start=1):
                # Récupération propre de la section
                text = (
                    section.split("] ---", 1)[-1].strip()
                    if "] ---" in section
                    else section
                )
                yield idx, text
        else:
            # Si c'est un fichier texte continu, découpage par blocs raisonnables (ex: ~2000 caractères)
            chunk_size = 2000
            for idx, start_pos in enumerate(
                range(0, len(content), chunk_size), start=1
            ):
                yield idx, content[start_pos : start_pos + chunk_size]

    else:
        # Fallback générique pour les autres formats (lecture texte)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
            if text:
                yield 1, text


# Dans app/ingestion/converter.py


def convert_and_reformat_document(
    file_path: Path, filename: str, tenant_id: str, pages_per_chunk: int = 3
) -> Generator[str, None, None]:
    """
    Version optimisée : Regroupe les pages par batch et utilise un cache/héritage
    de métadonnées pour minimiser les appels séquentiels au LLM.
    """
    file_path = Path(file_path)
    all_pages = list(extract_pages_or_chunks(file_path))

    if not all_pages:
        return

    last_known_meta = {
        "module": "Documentation Générale",
        "roles": "all",
        "screen": "Page Web / Documentation",
    }

    # Regroupement par lots (batching) pour réduire le nombre d'appels LLM
    for i in range(0, len(all_pages), pages_per_chunk):
        batch = all_pages[i : i + pages_per_chunk]
        batch_text = "\n\n".join([text for _, text in batch])
        start_idx = batch[0][0]
        end_idx = batch[-1][0]

        # Appel LLM uniquement si le texte du batch est suffisamment consistant
        if len(batch_text.strip()) > 50:
            try:
                meta = analyze_page_with_llm(
                    batch_text[:3000]
                )  # On limite la taille pour l'analyse rapide
                if meta and isinstance(meta, dict):
                    last_known_meta.update(
                        {
                            "module": meta.get("module") or last_known_meta["module"],
                            "roles": meta.get("roles") or last_known_meta["roles"],
                            "screen": meta.get("screen") or last_known_meta["screen"],
                        }
                    )
            except Exception as e:
                log.warning(
                    f"[Converter] Échec de l'analyse LLM pour le batch {start_idx}-{end_idx}, utilisation du contexte précédent : {e}"
                )

        # En-tête canonique unifié pour le bloc groupé
        formatted_header = (
            f"Tenant : {tenant_id}\n"
            f"Module : {last_known_meta['module']}\n"
            f"Rôles autorisés : {last_known_meta['roles']}\n"
            f"Écran : {last_known_meta['screen']}\n"
            f"URL : \n"
            f"---\n"
        )

        chunk_content = f"--- [Bloc Pages {start_idx} à {end_idx}] ---\n{batch_text}"
        yield f"{formatted_header}{chunk_content}"


""" 
def convert_and_reformat_document(
    file_path: Path, filename: str, tenant_id: str, pages_per_chunk: int
) -> Generator[str, None, None]:
    file_path = Path(file_path)

    with pdfplumber.open(str(file_path)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if not page_text or not page_text.strip():
                continue

            meta = analyze_page_with_llm(page_text)

            formatted_header = (
                f"Tenant : {tenant_id}\n"
                f"Module : {meta['module']}\n"
                f"Rôles autorisés : {meta['roles']}\n"
                f"Écran : {meta['screen']}\n"
                f"URL : \n"
                f"---\n"
            )

            chunk_content = f"--- [Page {idx}] ---\n{page_text}"
            yield f"{formatted_header}{chunk_content}"
"""
