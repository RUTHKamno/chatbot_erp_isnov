import io
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Generator

import pandas as pd
import pdfplumber
import requests
from PIL import Image
from pptx import Presentation

from app.ingestion import security
from app.ingestion.loaders.hf_models.vllm_model_def import (
    analyze_image_with_qwen,
    initialize_vlm_model,  # ou la fonction équivalente de votre fichier
)
from app.ingestion.loaders.hf_models.vlm_pipeline import (
    MIN_PAGE_CHARS,
    PAGE_PROMPT,
    _fallback_chunk,
    _run_vlm,
    check_grounding,
    clean_native_text,
    detect_boilerplate,
    ocr_page_regions,
    parse_structured,
)
from app.ingestion.settings import get_cfg

# processor, model = initialize_vlm_model()
log = logging.getLogger(__name__)


# ------------------------------------------------------------------ 1. EXTRACTION BRUTE (STREAMING / PAGE PAR PAGE)
"""
def stream_raw_text(
    file_path: Path, filename: str, pages_per_chunk: int = 5
) -> Generator[str, None, None]:
    
    #Lit le fichier et génère des morceaux de texte brut (chunks)
    #afin de pouvoir traiter les documents très volumineux (ex: 1000+ pages).
    
    file_path = Path(file_path)
    source_type = security.detect_source_type(file_path, filename)

    if source_type == "pdf":
        with pdfplumber.open(str(file_path)) as pdf:
            buffer = []
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    buffer.append(f"--- [Page {i}] ---\n{text}")
                # gestion des pdf avec images / éléments non textuels : on peut ajouter une analyse OCR si nécessaire
                if page.images:
                    # Rendu de la page entière en image pour préserver le contexte visuel global
                    pil_page_image = page.to_image(resolution=200).original
                    vlm_analysis = analyze_image_with_qwen(
                        pil_page_image, processor, model
                    )
                    if vlm_analysis:
                        buffer.append(
                            f"--- [Analyse Image Page {i}] ---\n{vlm_analysis}"
                        )

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
                if (
                    getattr(shape, "has_text_frame", False) and shape.has_text_frame
                ):  # noqa: SIM102
                    if shape.text_frame.text.strip():
                        slide_texts.append(shape.text_frame.text.strip())
            if slide_texts:
                buffer.append(f"--- [Diapositive {i}] ---\n" + "\n".join(slide_texts))
            # gestion des slides avec images : on peut ajouter une analyse OCR si nécessaire
            if slide.shapes:
                for shape in slide.shapes:
                    if (
                        getattr(shape, "image", None) or shape.shape_type == 1
                    ):  # Vérifie si c'est une image ou un type Picture:
                        image_stream = io.BytesIO(shape.image.blob)
                        pil_image = Image.open(image_stream)
                        vlm_analysis = analyze_image_with_qwen(
                            pil_image, processor, model
                        )
                        if vlm_analysis:
                            buffer.append(
                                f"--- [Analyse Image Diapositive {i}] ---\n{vlm_analysis}"
                            )

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
"""

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


""" ANICIENNE FONCTION (PAGE PAR PAGE) :
def extract_pages_or_chunks(file_path: Path) -> Generator[tuple[int, str], None, None]:
    
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
"""


def extract_pages_or_chunks(
    file_path: Path, processor=None, model=None, filename: str = None
) -> Generator[tuple[int, str], None, None]:
    """
    Générateur polymorphe qui extrait le texte page par page (PDF, PPTX)
    avec analyse visuelle des images par un modèle VLM, ou section par section (TXT, CSV, etc.).
    """
    ext = file_path.suffix.lower()
    doc_title = re.sub(r"[_\s]+", " ", Path(filename).stem).strip()

    if ext == ".pdf":
        with pdfplumber.open(str(file_path)) as pdf:
            print(
                f"🚀 Début de l'analyse du PDF {filename} ({len(pdf.pages)} pages au total) avec VLM..."
            )
            boilerplate = detect_boilerplate(pdf)
            print(f"🔍 Boilerplate détecté : {len(boilerplate)} motifs exclus.")

            for idx, page in enumerate(pdf.pages, start=1):
                print(f"\n----------------------------------------")
                print(f"📄 Traitement de la page {idx}/{len(pdf.pages)}...")

                native = clean_native_text(page, boilerplate)
                ocr_lines = ocr_page_regions(page, native)
                page_parts = []

                # 1. Évaluation du contenu utile
                total_chars = len(native) + sum(len(l) for l in ocr_lines)
                print(
                    f"📊 [Page {idx}] Caractères natifs : {len(native)} | Lignes OCR : {len(ocr_lines)} | Total utiles : {total_chars}"
                )

                if total_chars < MIN_PAGE_CHARS:
                    print(
                        f"⏩ [Page {idx}] Ignorée (en dessous du seuil MIN_PAGE_CHARS = {MIN_PAGE_CHARS})."
                    )
                    log.info(
                        "Page %s ignorée (%d caractères utiles).", idx, total_chars
                    )
                    continue

                # Page 100 % texte : le texte natif suffit, inutile d'appeler le VLM
                if not ocr_lines:
                    print(
                        f"📝 [Page {idx}] 100% texte détecté (pas de zones OCR/visuelles). Le texte natif suffit."
                    )
                    # Note: Si tu veux quand même yield le texte natif ici, assure-toi de l'ajouter à page_parts !
                    # Exemple : page_parts.append(native)
                    continue

                print(
                    f"🤖 [Page {idx}] Éléments visuels/OCR détectés. Préparation du prompt pour le VLM..."
                )
                prompt = PAGE_PROMPT.format(
                    doc_title=doc_title,
                    page=idx,
                    native=native or "(aucun)",
                    ocr="\n".join(ocr_lines),
                )

                page_text = page.extract_text()
                if page_text and page_text.strip():
                    page_parts.append(page_text.strip())

                # 2. Analyse des images / éléments visuels via le VLM si présent
                # if page.images and processor and model:
                if page.images:
                    print(
                        f"🖼️ [Page {idx}] {len(page.images)} image(s) trouvée(s) sur la page. Lancement de l'analyse VLM..."
                    )
                    try:
                        pil_page_image = page.to_image(resolution=150).original
                        print(f"⏳ [Page {idx}] Exécution de _run_vlm en cours...")
                        raw = _run_vlm(pil_page_image, prompt)
                        print(
                            f"📥 [Page {idx}] Sortie brute VLM reçue (longueur : {len(raw)} caractères) : \n{raw[:300]}..."
                        )  # Affiche un extrait

                        parsed = parse_structured(raw)
                        print(f"⚙️ [Page {idx}] Parsing structuré effectué.")

                        if not parsed.get("ETAPES") or len(raw) < 80:
                            print(
                                f"⚠️ [Page {idx}] Avertissement : Sortie VLM non structurée ou trop courte."
                            )
                            raise ValueError("sortie VLM non structurée")

                        score = check_grounding(
                            parsed, native + "\n" + "\n".join(ocr_lines)
                        )
                        print(
                            f"🎯 [Page {idx}] Score de grounding calculé : {score:.2f}"
                        )

                        if score < 0.3:
                            print(
                                f"⚠️ [Page {idx}] Attention : Faible correspondance des libellés dans l'OCR (score {score:.2f})."
                            )
                            log.warning(
                                "Page %s : peu de libellés retrouvés dans l'OCR (%.2f) → à relire",
                                idx,
                                score,
                            )

                        # Attention ici : parsed est un objet/dict ou une chaîne ? S'il s'agit d'une string, `.strip()` fonctionne, sinon adapte selon ton type de retour `parsed`
                        parsed_str = (
                            parsed.strip() if isinstance(parsed, str) else str(parsed)
                        )
                        page_parts.append(f"[Page {idx} :\n{parsed_str}]")
                        print(f"✅ [Page {idx}] Analyse VLM intégrée avec succès.")

                    except Exception as e:  # noqa: BLE001
                        print(f"❌ [Page {idx}] ERREUR lors de l'analyse VLM : {e}")
                else:
                    print(
                        f"ℹ️ [Page {idx}] Pas d'appel VLM (Soit pas d'images sur la page, soit processeur/modèle non chargés)."
                    )

                if page_parts:
                    combined_content = "\n\n".join(page_parts)
                    print(
                        f"📤 [Page {idx}] Yield du bloc final (longueur : {len(combined_content)} caractères)."
                    )
                    yield idx, combined_content
                else:
                    print(
                        f"⚠️ [Page {idx}] Aucun contenu généré pour cette page (page_parts vide)."
                    )
    elif ext == ".pptx":
        prs = Presentation(str(file_path))
        for idx, slide in enumerate(prs.slides, start=1):
            slide_parts = []

            # 1. Extraction du texte des formes de la diapositive
            slide_texts = []
            for shape in slide.shapes:
                if (
                    getattr(shape, "has_text_frame", False) and shape.has_text_frame
                ):  # noqa: SIM102
                    if shape.text_frame.text.strip():
                        slide_texts.append(shape.text_frame.text.strip())
            if slide_texts:
                slide_parts.append("\n".join(slide_texts))

            # 2. Extraction et analyse des images intégrées dans la diapositive via le VLM
            if processor and model:
                for shape in slide.shapes:
                    if (
                        getattr(shape, "image", None)
                        or getattr(shape, "shape_type", None) == 1
                    ):
                        try:
                            image_stream = io.BytesIO(shape.image.blob)
                            pil_image = Image.open(image_stream)
                            vlm_analysis = analyze_image_with_qwen(
                                pil_image, processor, model
                            )
                            if vlm_analysis and vlm_analysis.strip():
                                slide_parts.append(
                                    f"[Analyse Image Diapositive {idx} :\n{vlm_analysis.strip()}]"
                                )
                        except Exception:  # noqa: BLE001, S112
                            continue

            if slide_parts:
                combined_slide = "\n\n".join(slide_parts)
                yield idx, combined_slide

    elif ext in (".txt", ".md", ".log"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()

        sections = [s.strip() for s in content.split("--- [Page ") if s.strip()]
        if len(sections) > 1:
            for idx, section in enumerate(sections, start=1):
                text = (
                    section.split("] ---", 1)[-1].strip()
                    if "] ---" in section
                    else section
                )
                yield idx, text
        else:
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
    file_path: Path,
    filename: str,
    tenant_id: str,
    pages_per_chunk: int = 3,
    cancel_check: Callable[[], bool] | None = None,
) -> Generator[str, None, None]:
    """
    Version optimisée : Regroupe les pages par batch et utilise un cache/héritage
    de métadonnées pour minimiser les appels séquentiels au LLM.
    """
    file_path = Path(file_path)
    all_pages = list(extract_pages_or_chunks(file_path, filename=filename))

    if not all_pages:
        return

    last_known_meta = {
        "module": "Documentation Générale",
        "roles": "all",
        "screen": "Page Web / Documentation",
    }

    # Regroupement par lots (batching) pour réduire le nombre d'appels LLM
    for i in range(0, len(all_pages), pages_per_chunk):
        if cancel_check and cancel_check():
            return

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
        print(
            f"📤 Yield du bloc Pages {start_idx}-{end_idx} avec en-tête unifié et contenu de longueur {len(batch_text)} caractères."
        )
        print(
            f"---\n{formatted_header}{batch_text[:500]}...\n---"
        )  # Affiche un extrait du contenu

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
