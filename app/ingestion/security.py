"""Validation des entrées : identifiants, type réel des fichiers, détection d'injection de prompt."""
import re
import zipfile
from pathlib import Path
from typing import List

TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_]{1,39}$")
ROLE_RE = re.compile(r"^[a-z0-9_\-]{1,40}$")
USER_RE = re.compile(r"^[A-Za-z0-9_\-\.@]{1,100}$")
DOC_RE = re.compile(r"^[0-9a-f]{32}$")
SLUG_RE = re.compile(r"^[a-z0-9_\-]{1,40}$")


class IsolationError(RuntimeError):
    """Un document d'une autre entreprise a été détecté dans un résultat : on bloque tout."""


class UnsupportedFile(ValueError):
    pass


def check_tenant(t, allow_global: bool = False) -> str:
    t = (t or "").strip() if isinstance(t, str) else ""
    if t == "global" and allow_global:
        return t
    if not TENANT_RE.match(t) or t == "global":
        raise ValueError("tenant_id invalide (minuscules, chiffres et _ ; 2 à 40 caractères ; 'global' réservé)")
    return t


def check_role(r) -> str:
    r = (r or "").strip().lower() if isinstance(r, str) else ""
    if not ROLE_RE.match(r):
        raise ValueError("rôle invalide")
    return r


def check_user(u) -> str:
    u = (u or "").strip() if isinstance(u, str) else ""
    if not USER_RE.match(u):
        raise ValueError("identifiant utilisateur invalide")
    return u


def check_doc_id(d) -> str:
    d = (d or "").strip() if isinstance(d, str) else ""
    if not DOC_RE.match(d):
        raise ValueError("doc_id invalide")
    return d


def check_slug(s, label: str = "valeur") -> str:
    s = (s or "").strip().lower() if isinstance(s, str) else ""
    if not SLUG_RE.match(s):
        raise ValueError(f"{label} invalide (lettres minuscules, chiffres, - et _)")
    return s


# ------------------------------------------------------------------ type réel du fichier
TEXT_EXTS = {".txt": "text", ".md": "text", ".csv": "csv", ".tsv": "csv"}
NOT_YET = {
    "image": "Les images seront prises en charge en phase 2 (OCR / vision).",
    "audio": "Les fichiers audio seront pris en charge en phase 3 (transcription).",
    "docx": "Le format Word (.docx) n'est pas encore pris en charge.",
    "xlsx": "Le format Excel (.xlsx) n'est pas encore pris en charge : exportez en CSV.",
}


def detect_source_type(path: Path, filename: str) -> str:
    """Détermine le type RÉEL via les octets magiques, puis vérifie la cohérence avec l'extension."""
    ext = Path(filename).suffix.lower()
    with open(path, "rb") as f:
        head = f.read(16)
        f.seek(0)
        sample = f.read(8192)

    if head.startswith(b"%PDF-"):
        detected = "pdf"
    elif head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
        except zipfile.BadZipFile:
            raise UnsupportedFile("Fichier corrompu (archive illisible).")
        if "ppt/presentation.xml" in names:
            detected = "pptx"
        elif "word/document.xml" in names:
            detected = "docx"
        elif "xl/workbook.xml" in names:
            detected = "xlsx"
        else:
            raise UnsupportedFile("Archive ZIP non supportée.")
    elif head.startswith((b"\xff\xd8\xff", b"\x89PNG", b"GIF8")):
        detected = "image"
    elif head.startswith((b"RIFF", b"ID3", b"OggS", b"fLaC")) or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        detected = "audio"
    elif b"\x00" in sample:
        raise UnsupportedFile("Fichier binaire non reconnu.")
    else:
        if ext not in TEXT_EXTS:
            raise UnsupportedFile(f"Extension '{ext}' non supportée (attendu : .pdf .pptx .csv .txt .md).")
        return TEXT_EXTS[ext]

    if detected in NOT_YET:
        raise UnsupportedFile(NOT_YET[detected])
    expected_ext = {"pdf": {".pdf"}, "pptx": {".pptx"}}[detected]
    if ext not in expected_ext:
        raise UnsupportedFile(f"Le contenu est de type {detected} mais l'extension est '{ext}' : fichier rejeté.")
    return detected


# ------------------------------------------------------------------ injection de prompt
_INJECTION = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any )?(the )?(previous|prior|above) ",
    r"ignore[zr]? (toutes )?(les )?(consignes|instructions|règles) (précédentes|ci-dessus|antérieures)",
    r"oublie[zr]? (toutes )?(les )?(consignes|instructions) ",
    r"\byou are now\b",
    r"\b(tu es|vous êtes) maintenant\b",
    r"reveal (the |your )?system prompt",
    r"(révèle|affiche|montre|donne)[a-z]* (moi )?(le |ton |votre )?(prompt|message) syst[eè]me",
    r"<\s*/?\s*(system|assistant)\s*>",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION]


def scan_injection(text: str) -> List[str]:
    """Renvoie les extraits suspects (liste vide si rien). Un document suspect part en revue manuelle."""
    hits = []
    for rx in _INJECTION_RE:
        m = rx.search(text)
        if m:
            hits.append(m.group(0)[:80])
    return hits
