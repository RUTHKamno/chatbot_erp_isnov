"""
Pipeline PDF « documentation avec captures d'écran » → chunks prêts pour un RAG.

Pour chaque page :
  1. texte natif (pdfplumber)            → titres, légendes (fiable)
  2. OCR des zones d'images (Tesseract)  → libellés des boutons/colonnes DANS les captures
  3. VLM (page entière + 1 + 2 en indices) → description structurée de la procédure
  4. un chunk par page, avec métadonnées (module, écran, rôles) tirées du VLM
     (plus besoin d'un second appel LLM pour les métadonnées).

Pages sans contenu utile (couverture, remerciements) : ignorées.
"""

import logging
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Generator

import pytesseract
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

# Windows : chemin habituel de Tesseract si non présent dans le PATH
_WIN_TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
elif os.name == "nt" and Path(_WIN_TESS).exists():
    pytesseract.pytesseract.tesseract_cmd = _WIN_TESS
else:
    # Sous Linux (Ubuntu), Tesseract est généralement accessible globalement via le PATH (/usr/bin/tesseract)
    # On peut laisser pytesseract utiliser la commande par défaut "tesseract"
    pass

OCR_LANG = "fra+eng"
OCR_MIN_CONF = 55
MIN_PAGE_CHARS = (
    250  # texte natif + OCR : en dessous, page ignorée (couverture, remerciements)
)
MIN_TEXT_ONLY_CHARS = (
    150  # page sans capture : seuil pour la traiter comme page de texte
)


# ============================================================ 1. OCR
def _group_words_into_lines(words: list[tuple]) -> list[str]:
    """words = (left, top, width, height, text) → lignes dans l'ordre de lecture.
    Une ligne est coupée en segments quand l'écart horizontal est grand
    (évite de coller un libellé de gauche avec une légende de droite)."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (w[1] + w[3] / 2, w[0]))
    rows, cur = [], [words[0]]
    for w in words[1:]:
        cy = sum(x[1] + x[3] / 2 for x in cur) / len(cur)
        h = sum(x[3] for x in cur) / len(cur)
        if abs((w[1] + w[3] / 2) - cy) <= 0.6 * h:
            cur.append(w)
        else:
            rows.append(cur)
            cur = [w]
    rows.append(cur)

    lines = []
    for row in rows:
        row.sort(key=lambda w: w[0])
        h = sum(x[3] for x in row) / len(row)
        seg, last_right = [row[0][4]], row[0][0] + row[0][2]
        for x in row[1:]:
            if x[0] - last_right > 2.5 * h:
                lines.append(" ".join(seg))
                seg = [x[4]]
            else:
                seg.append(x[4])
            last_right = x[0] + x[2]
        lines.append(" ".join(seg))
    return lines


def _is_noise(line: str) -> bool:
    """Écarte les lignes OCR qui ressemblent à du bruit (photos, icônes, traits)."""
    if len(line) < 2:
        return True
    alnum = sum(c.isalnum() for c in line)
    return alnum / max(len(line), 1) < 0.5


def _fold(s: str) -> str:
    """minuscules, sans accents ni ponctuation : sert au dédoublonnage et aux comparaisons."""
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(
        r"[^a-z0-9]", "", "".join(c for c in s if not unicodedata.combining(c))
    )


def _ocr_pass(gray: Image.Image, psm: int) -> list[str]:
    data = pytesseract.image_to_data(
        gray,
        lang=OCR_LANG,
        config=f"--oem 3 --psm {psm}",
        output_type=pytesseract.Output.DICT,
    )
    words = []
    for i, txt in enumerate(data["text"]):
        txt = txt.strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf >= OCR_MIN_CONF:
            words.append(
                (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                    txt,
                )
            )
    return [l for l in _group_words_into_lines(words) if not _is_noise(l)]


def ocr_image(img: Image.Image) -> list[str]:
    """OCR d'une capture : deux passes complémentaires fusionnées.
    - niveaux de gris, psm 6 (blocs de texte : tableaux, formulaires)
    - canal rouge, psm 11 (texte épars : boutons sombres sur fond orange/jaune)
    Mesuré sur un PDF iNOV : le canal rouge récupère des libellés que le gris rate.
    Pas d'agrandissement : à 300 dpi, x2 dégrade les accents."""
    rgb = img.convert("RGB")
    passes = [
        (ImageOps.autocontrast(ImageOps.grayscale(rgb)), 6),
        (ImageOps.autocontrast(rgb.split()[0]), 11),
    ]
    lines, seen = [], set()
    for gray, psm in passes:
        if max(gray.size) < 1500:  # petite zone : on agrandit
            gray = gray.resize((gray.width * 2, gray.height * 2), Image.LANCZOS)
        for line in _ocr_pass(gray, psm):
            key = _fold(line)
            if key and key not in seen:
                seen.add(key)
                lines.append(line)
    return lines


# ============================================================ 2. PAGES PDF
def _norm(line: str) -> str:
    """Normalisation pour comparer les lignes d'en-tête/pied entre pages
    (ignore espaces, casse et numéros de page : 'iNOVFAQ' == 'iNOV FAQ')."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", "", line.lower()))


def detect_boilerplate(pdf, ratio: float = 0.3, min_pages: int = 4) -> set[str]:
    """Lignes qui reviennent sur beaucoup de pages (pied de page, coordonnées...)."""
    n = len(pdf.pages)
    if n < min_pages:
        return set()
    counter: Counter = Counter()
    for p in pdf.pages:
        counter.update(
            {_norm(l) for l in (p.extract_text() or "").splitlines() if l.strip()}
        )
    threshold = max(2, ratio * n)
    return {line for line, c in counter.items() if c >= threshold}


def clean_native_text(page, boilerplate: set[str]) -> str:
    lines = (page.extract_text() or "").splitlines()
    return "\n".join(
        l.strip() for l in lines if l.strip() and _norm(l) not in boilerplate
    )


def content_image_boxes(page, min_w: float = 120, min_h: float = 60) -> list[tuple]:
    """Zones d'images assez grandes pour être des captures (écarte logos/icônes)."""
    x0p, topp, x1p, botp = page.bbox
    boxes = []
    for im in page.images:
        if (im["x1"] - im["x0"]) < min_w or (im["bottom"] - im["top"]) < min_h:
            continue
        boxes.append(
            (
                max(im["x0"], x0p),
                max(im["top"], topp),
                min(im["x1"], x1p),
                min(im["bottom"], botp),
            )
        )
    return boxes


def ocr_page_regions(page, native_text: str) -> list[str]:
    boxes = content_image_boxes(page)
    if not boxes and len(native_text) < MIN_TEXT_ONLY_CHARS:
        boxes = [page.bbox]  # page scannée ou captures dessinées en vectoriel
    lines: list[str] = []
    for box in boxes:
        try:
            crop = page.crop(box).to_image(resolution=300).original
            lines.extend(ocr_image(crop))
        except Exception as e:  # noqa: BLE001
            log.warning("OCR échoué sur une zone (page %s) : %s", page.page_number, e)
    return lines


# ============================================================ 3. VLM
PAGE_PROMPT = """You are analyzing ONE page of an ERP user documentation (annotated screenshot(s) and captions).
Your output will be indexed in a vector database (RAG) to answer employee queries: it must be autonomous, factual, and written in French.
 
Document: {doc_title}
Page: {page}
 
NATIVE PDF TEXT (reliable: titles and captions):
<<<
{native}
>>>
 
OCR TEXT READ FROM SCREENSHOTS (may contain reading errors; use it to find the exact labels of buttons, tabs, fields, columns, and values):
<<<
{ocr}
>>>
 
Document conventions:
- Numbered circles (1, 2, 3...) indicate the ORDER of steps.
- A colored rectangle (pink/purple) surrounds the target element to click; an arrow connects it to its caption.
- A dashed arrow indicates the screen or window that opens after clicking.
- Ignore: logos, diagonal watermarks, footers, and company contact details.
 
Strict rules:
- Do not invent anything: describe only what is visible or written. If illegible, write "illisible".
- Use EXACT labels enclosed in French guillemets « » (buttons, tabs, fields, columns).
- Each step must specify: WHERE (screen/tab), WHAT (action on which element), and the RESULT obtained.
- If a table is visible, cite its column headers and provide 1 to 2 example rows.
- If a dashed arrow points to a new window, explicitly describe the resulting window or state change in the step.
 
Respond EXACTLY with this format, one line per field (or block for ETAPES), without markdown formatting:
TYPE: <procedure | referentiel | autre>
TITRE: <title of the procedure>
MODULE: <ERP module, e.g., Ventes, Comptabilité, Stocks>
ECRAN: <windows, tabs, or menu paths involved>
ROLES: <roles involved, or all if not specified>
OBJECTIF: <what the procedure achieves, 1 to 2 sentences>
ETAPES:
1. <step>
2. <step>
ELEMENTS_UI: <important buttons, fields, columns with their exact labels>
A_SAVOIR: <conditions, effects, technical notes; otherwise none>
"""

FIELDS = [
    "TYPE",
    "TITRE",
    "MODULE",
    "ECRAN",
    "ROLES",
    "OBJECTIF",
    "ETAPES",
    "ELEMENTS_UI",
    "A_SAVOIR",
]
_FIELD_RE = re.compile(r"^(%s)\s*:\s*" % "|".join(FIELDS), re.M)


def _run_vlm(image: Image.Image, prompt: str) -> str:
    print("Appel du VLM pour analyser la page...in _run_vlm")
    # import paresseux : le modèle n'est chargé qu'au premier appel
    from app.ingestion.loaders.hf_models.vllm_model_def import (
        analyze_image_with_qwen,
        initialize_vlm_model,
    )

    init_model = (
        initialize_vlm_model()
    )  # Assure que le modèle est chargé avant l'analyse

    return analyze_image_with_qwen(
        image, init_model[0], init_model[1], prompt, PROVIDER="ollama"
    )  # processor, model


def parse_structured(text: str) -> dict[str, str]:
    print("Parsing structured output from VLM...")
    matches = list(_FIELD_RE.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[m.group(1)] = text[m.end() : end].strip()
    return out


def check_grounding(parsed: dict[str, str], source_text: str) -> float:
    """Part des libellés « entre guillemets » cités par le VLM retrouvés dans le
    texte natif + OCR. Indicateur de contrôle (pas un verdict) : l'OCR rate certains
    boutons que seul le VLM lit, donc un score bas = « à relire », pas forcément faux.
    """
    labels = re.findall(r"«\s*([^»]{2,60}?)\s*»", " ".join(parsed.values()))
    if not labels:
        return 1.0
    src = _fold(source_text)
    ok = sum(1 for l in labels if _fold(l) in src)
    return ok / len(labels)


def _fallback_chunk(tenant_id, doc_title, page_no, native, ocr_lines) -> str:
    """Si le VLM échoue : on garde au moins texte natif + OCR (rien n'est perdu)."""
    header = (
        f"Tenant : {tenant_id}\nModule : Non spécifié\nRôles autorisés : all\n"
        f"Écran : Non spécifié\nURL : \n---\n"
    )
    body = f"Document : {doc_title}\nPage : {page_no}\n{native}"
    if ocr_lines:
        body += "\nTexte des captures (OCR) :\n" + "\n".join(ocr_lines)
    return header + body
