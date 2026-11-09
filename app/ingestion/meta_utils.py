"""Lecture des métadonnées « Clé : valeur » écrites DANS les documents. Partagé par tous les loaders."""
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

_ALIASES = {
    "tenant": "tenant_id", "tenant_id": "tenant_id", "entreprise": "tenant_id",
    "roles autorises": "roles", "role autorise": "roles", "roles": "roles", "role": "roles",
    "module": "module", "ecran": "ecran", "url": "url",
    "type de document": "doc_type", "doc_type": "doc_type",
}
_KEY_RE = re.compile(r"^\s*([A-Za-zÀ-ÿ_' ]{2,30}?)\s*:\s*(.+?)\s*$")   # tolère « Tenant: x » et « Tenant : x »
_ALL = {"all", "tous", "tout", "*"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


def parse_meta_line(line: str) -> Optional[Tuple[str, str]]:
    m = _KEY_RE.match(line)
    if not m:
        return None
    key = _ALIASES.get(_norm(m.group(1)))
    return (key, m.group(2).strip()) if key else None


def parse_roles(value: str) -> List[str]:
    roles = []
    for r in re.split(r"[;,|]", value or ""):
        r = r.strip().lower()
        if r:
            roles.append("all" if _norm(r) in _ALL else r)
    return sorted(set(roles))


def extract_metadata(lines: List[str]) -> Tuple[Dict, List[str]]:
    """Sépare les lignes de métadonnées (dict ÉPARS : seulement les clés présentes) du contenu."""
    meta, rest = {}, []
    for line in lines:
        kv = parse_meta_line(line)
        if kv:
            key, val = kv
            meta[key] = parse_roles(val) if key == "roles" else val.strip()
        else:
            rest.append(line)
    return meta, rest