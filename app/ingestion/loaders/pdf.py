import re
from collections import Counter
from pathlib import Path

import pdfplumber

from app.ingestion.meta_utils import parse_meta_line, parse_roles
from app.ingestion.models import Block, ExtractedDocument

MODULE_HEADING_RE = re.compile(r"^\d+\. Module \S")      # intitulés de chapitres du guide ERP
PAGE_NUMBER_RE = re.compile(r"\bpage\b|\b\d+\s*/\s*\d+\b", re.IGNORECASE)    # « Page 3 », « 3 / 10 »


def _shape(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def _is_noise(line: str, repeated: set) -> bool:
    return line.strip() in repeated or (bool(PAGE_NUMBER_RE.search(line)) and _shape(line) in repeated)


def load_pdf(path: Path) -> ExtractedDocument:
    pages, empty_pages = [], 0
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            lines = [l.strip() for l in (page.extract_text() or "").splitlines() if l.strip()]
            if not lines:
                empty_pages += 1
            pages.append(lines)
    total_pages = len(pages)

    # En-têtes / pieds de page : lignes répétées en haut ou en bas sur >= 60 % des pages (jamais les métadonnées)
    repeated = set()
    if total_pages >= 3:
        exact, shaped = Counter(), Counter()
        for lines in pages:
            for l in set(lines[:2] + lines[-2:]):
                if not parse_meta_line(l):
                    exact[l] += 1
                    if PAGE_NUMBER_RE.search(l):
                        shaped[_shape(l)] += 1
        need = max(3, 0.6 * total_pages)
        repeated = {k for k, c in exact.items() if c >= need} | {k for k, c in shaped.items() if c >= need}

    blocks, defaults, cur, paras, section_page, preamble = [], {}, None, [], None, []

    def close_section():
        if cur is not None and paras:
            blocks.append(Block("\n".join(paras).strip(), {**cur, "page": section_page}))

    for page_no, lines in enumerate(pages, 1):
        for line in lines:
            if _is_noise(line, repeated) or MODULE_HEADING_RE.match(line):
                continue
            kv = parse_meta_line(line)
            if kv:
                key, val = kv
                if key == "ecran":                      # « Écran : » démarre une nouvelle section
                    close_section()
                    cur, paras, section_page = {**defaults, "ecran": val}, [], page_no   # hérite des défauts du DOCUMENT seulement
                else:                                   # avant la 1re section => valeurs par défaut du document
                    (cur if cur is not None else defaults)[key] = parse_roles(val) if key == "roles" else val
                continue
            if cur is None:
                preamble.append((page_no, line))        # page de garde / document sans sections
            else:
                paras.append(line)
    close_section()

    # Document sans « Écran : » (cours, note...) : un bloc par page, avec les métadonnées du document
    if not blocks and preamble:
        by_page = {}
        for p, l in preamble:
            by_page.setdefault(p, []).append(l)
        blocks = [Block("\n".join(ls), {**defaults, "page": p}) for p, ls in by_page.items()]

    warnings = []
    if empty_pages:
        warnings.append(f"{empty_pages}/{total_pages} page(s) sans texte extractible (scan ? l'OCR arrive en phase 2).")
    return ExtractedDocument(
        blocks=blocks, warnings=warnings,
        stats={"pages_total": total_pages, "pages_without_text": empty_pages, "sections_extracted": len(blocks)},
        partial=bool(empty_pages and blocks),
    )