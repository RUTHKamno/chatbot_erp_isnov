import re
from pathlib import Path
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from app.ingestion.models import Block, ExtractedDocument


def _shape_texts(shape, out: list, counters: dict):
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for s in shape.shapes:
            _shape_texts(s, out, counters)
        return
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        counters["images"] += 1
        return
    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        t = shape.text_frame.text.strip()
        if t:
            out.append(t)
    if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                out.append(" | ".join(cells))


def parse_metadata_from_text(lines: list) -> dict:
    """Analyse les lignes d'un bloc à la recherche de clés de métadonnées."""
    meta = {"roles": ["all"], "tenant_id": "default", "module": "général", "ecran": "", "url": ""}
    clean_lines = []

    for line in lines:
        if line.startswith("Tenant : "):
            meta["tenant_id"] = line[9:].strip() or "default"
        elif line.startswith("Module : "):
            meta["module"] = line[9:].strip() or "général"
        elif line.startswith("Rôles autorisés : "):
            parsed = [r.strip() for r in line[18:].split(",") if r.strip()]
            meta["roles"] = parsed if parsed else ["all"]
        elif line.startswith("Écran : "):
            meta["ecran"] = line[8:].strip()
        elif line.startswith("URL : "):
            meta["url"] = line[6:].strip()
        else:
            clean_lines.append(line)

    return meta, "\n".join(clean_lines).strip()


def load_pptx(path: Path) -> ExtractedDocument:
    prs = Presentation(str(path))
    counters, empty, slides = {"images": 0}, 0, []

    for n, slide in enumerate(prs.slides, 1):
        parts = []
        for shape in slide.shapes:
            _shape_texts(shape, parts, counters)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(notes)
        if not parts:
            empty += 1
            continue

        title = None
        if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
            title = slide.shapes.title.text_frame.text.strip() or None

        # IMPORTANT : une zone de texte contient plusieurs lignes -> on éclate avant de chercher les métadonnées
        lines = [l.strip() for p in parts for l in p.replace("\x0b", "\n").splitlines() if l.strip()]
        meta, rest = extract_metadata(lines)
        slides.append((n, title, meta, rest))

    # Les métadonnées de la 1re diapositive qui en déclare valent pour tout le document ;
    # chaque diapositive peut les surcharger. Aucun héritage d'une diapositive à l'autre.
    defaults = next((m for _, _, m, _ in slides if m), {})
    blocks = []
    for n, title, meta, rest in slides:
        full = {**defaults, **meta}
        if not full.get("ecran") and title:
            full["ecran"] = title
        text = "\n".join(rest).strip()
        if text:
            blocks.append(Block(text, {**full, "slide": n}))

    return ExtractedDocument(
        blocks=blocks,
        warnings=[f"{counters['images']} image(s) non analysée(s)."] if counters["images"] else [],
        stats={"slides_total": len(prs.slides), "slides_empty": empty},
    )