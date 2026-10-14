import re
from pathlib import Path

from app.ingestion.models import Block, ExtractedDocument

_HEADING = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _decode(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_metadata_from_text(text: str) -> tuple[dict, str]:
    meta = {"roles": ["all"], "tenant_id": "default", "module": "général", "ecran": "", "url": ""}
    clean_lines = []
    
    for line in text.splitlines():
        line_str = line.strip()
        if line_str.startswith("Tenant : "):
            meta["tenant_id"] = line_str[9:].strip() or "default"
        elif line_str.startswith("Module : "):
            meta["module"] = line_str[9:].strip() or "général"
        elif line_str.startswith("Rôles autorisés : "):
            parsed = [r.strip() for r in line_str[18:].split(",") if r.strip()]
            meta["roles"] = parsed if parsed else ["all"]
        elif line_str.startswith("Écran : "):
            meta["ecran"] = line_str[8:].strip()
        elif line_str.startswith("URL : "):
            meta["url"] = line_str[6:].strip()
        else:
            clean_lines.append(line)

    return meta, "\n".join(clean_lines).strip()


def load_text(path: Path) -> ExtractedDocument:
    raw_text = _decode(path).strip()
    if not raw_text:
        return ExtractedDocument(blocks=[], warnings=["Fichier vide."])

    heads = list(_HEADING.finditer(raw_text)) if path.suffix.lower() == ".md" else []

    # Métadonnées du DOCUMENT = lignes « Clé : valeur » placées avant le 1er titre (ou dans tout le fichier sans titres)
    head_end = heads[0].start() if heads else len(raw_text)
    defaults, preamble_lines = extract_metadata(raw_text[:head_end].splitlines())
    preamble = "\n".join(preamble_lines).strip()

    if not heads:
        blocks = [Block(preamble, dict(defaults))] if preamble else []
        return ExtractedDocument(blocks=blocks, stats={"chars": len(preamble)})

    blocks = [Block(preamble, dict(defaults))] if preamble else []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(raw_text)
        meta, lines = extract_metadata(raw_text[m.start():end].splitlines())
        full = {**defaults, **meta}                 # la section hérite des défauts du document et peut les surcharger
        if not full.get("ecran"):
            full["ecran"] = m.group(2).strip()
        body = "\n".join(lines).strip()
        if body:
            blocks.append(Block(body, full))

    return ExtractedDocument(blocks=blocks, stats={"chars": len(raw_text), "sections": len(blocks)})