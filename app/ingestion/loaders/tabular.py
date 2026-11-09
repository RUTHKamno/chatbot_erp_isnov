import io
import re
import unicodedata
from pathlib import Path

import pandas as pd

from app.ingestion.meta_utils import extract_metadata, parse_roles
from app.ingestion.models import Block, ExtractedDocument
from app.ingestion.settings import get_cfg

# Colonnes interprétées comme métadonnées UNIQUEMENT dans un CSV « guide » (qui possède une colonne instructions)
_COLS = {
    "roles": "roles", "role": "roles", "roles_autorises": "roles", "role_autorise": "roles",
    "tenant_id": "tenant_id", "tenant": "tenant_id", "entreprise": "tenant_id",
    "module": "module", "ecran": "ecran", "url": "url", "doc_type": "doc_type",
    "instructions": "instructions", "texte": "instructions", "description": "instructions",
}


def _col_key(name) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).strip().lower()
    return re.sub(r"[\s\-]+", "_", s)


def _read(path: Path):
    """Renvoie (DataFrame, lignes d'en-tête '# Clé : valeur' placées tout en haut du fichier)."""
    last = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            lines = path.read_text(encoding=enc).splitlines()
        except UnicodeDecodeError as e:
            last = e
            continue
        i, header = 0, []
        while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
            if lines[i].strip():
                header.append(lines[i].lstrip("# ").strip())
            i += 1
        body = "\n".join(lines[i:])
        for kwargs in ({"sep": None, "engine": "python"}, {"sep": ","}):
            try:
                return pd.read_csv(io.StringIO(body), **kwargs), header
            except Exception as e:
                last = e
    raise ValueError(f"CSV illisible : {last}")


def load_csv(path: Path) -> ExtractedDocument:
    cfg = get_cfg()
    df, header_lines = _read(path)
    total = len(df)
    if total == 0:
        return ExtractedDocument(blocks=[], warnings=["Tableau vide."], stats={"rows_total": 0})

    warnings, partial = [], False
    if total > cfg.csv_max_rows:
        df = df.head(cfg.csv_max_rows)
        warnings.append(f"Tableau tronqué aux {cfg.csv_max_rows} premières lignes sur {total}.")
        partial = True

    defaults, _ = extract_metadata(header_lines)        # « # Tenant : acme », « # Rôles autorisés : rh » en tête de fichier
    cols = {}
    for c in df.columns:
        cols.setdefault(_COLS.get(_col_key(c)), c)

    blocks = []
    if "instructions" in cols:
        # ---- MODE GUIDE (comme ingest_guide.py) : une ligne = une section avec ses propres métadonnées
        def cell(row, key):
            if key not in cols:
                return ""
            v = row[cols[key]]
            return "" if pd.isna(v) else str(v).strip()

        for n, (_, row) in enumerate(df.iterrows(), 1):
            text = cell(row, "instructions")
            if not text:
                continue
            meta = dict(defaults)
            for key in ("tenant_id", "module", "ecran", "url", "doc_type"):
                if cell(row, key):
                    meta[key] = cell(row, key)
            if cell(row, "roles"):
                meta["roles"] = parse_roles(cell(row, "roles"))
            meta.update(row_start=n, row_end=n)
            blocks.append(Block(text, meta))
    else:
        # ---- MODE TABLEAU DE DONNÉES : groupes de lignes (1 groupe = 1 chunk) ; métadonnées = en-tête '#' du fichier
        header = f"Tableau « {path.stem} » — colonnes : " + ", ".join(map(str, df.columns))
        budget = max(200, cfg.chunk_size - len(header) - 1)
        group, size, first = [], 0, 1

        def flush(last_row: int):
            nonlocal group, size, first
            if group:
                blocks.append(Block(header + "\n" + "\n".join(group),
                                    {**defaults, "row_start": first, "row_end": last_row}))
            group, size, first = [], 0, last_row + 1

        for n, (_, row) in enumerate(df.iterrows(), 1):
            line = " | ".join(f"{c}: {v}" for c, v in row.items() if pd.notna(v) and str(v).strip())[:budget]
            if group and (size + len(line) + 1 > budget or len(group) >= cfg.csv_rows_per_chunk):
                flush(n - 1)
            if line:
                group.append(line)
                size += len(line) + 1
        flush(len(df))

    return ExtractedDocument(
        blocks=blocks, warnings=warnings,
        stats={"rows_total": total, "columns": list(map(str, df.columns)), "mode": "guide" if "instructions" in cols else "tableau"},
        partial=partial,
    )