"""Registre SQLite des documents : statut, versions, droits. Toutes les lectures sont filtrées par tenant."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.ingestion import security
from app.ingestion.settings import get_cfg

STATUSES = ("pending", "processing", "indexed", "needs_review", "failed", "superseded", "deleted")

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL,
    source_name TEXT NOT NULL, source_type TEXT NOT NULL, content_hash TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL, status_detail TEXT DEFAULT '',
    n_chunks INTEGER DEFAULT 0, visibility TEXT NOT NULL, allowed_roles TEXT DEFAULT '',
    doc_type TEXT DEFAULT 'document', module TEXT DEFAULT '', tags TEXT DEFAULT '',
    file_path TEXT, warnings TEXT DEFAULT '[]', stats TEXT DEFAULT '{}',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_tenant ON documents (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents (tenant_id, owner_id, content_hash);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registry:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or (get_cfg().data_dir / "ingestion_registry.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_DDL)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _d(row) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        d = dict(row)
        d["allowed_roles"] = [r for r in d["allowed_roles"].split("|") if r]
        d["tags"] = [t for t in d["tags"].split("|") if t]
        d["warnings"] = json.loads(d["warnings"] or "[]")
        d["stats"] = json.loads(d["stats"] or "{}")
        return d

    def create(self, **f) -> None:
        f.setdefault("created_at", now())
        f["updated_at"] = f["created_at"]
        f["allowed_roles"] = "|".join(f.get("allowed_roles", []))
        f["tags"] = "|".join(f.get("tags", []))
        cols = ", ".join(f)
        marks = ", ".join("?" for _ in f)
        with self._conn() as c:
            c.execute(f"INSERT INTO documents ({cols}) VALUES ({marks})", list(f.values()))

    def update(self, doc_id: str, **f) -> None:
        if "warnings" in f:
            f["warnings"] = json.dumps(f["warnings"], ensure_ascii=False)
        if "stats" in f:
            f["stats"] = json.dumps(f["stats"], ensure_ascii=False, default=str)
        f["updated_at"] = now()
        sets = ", ".join(f"{k} = ?" for k in f)
        with self._conn() as c:
            c.execute(f"UPDATE documents SET {sets} WHERE doc_id = ?", [*f.values(), doc_id])

    def get_internal(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Pour le worker uniquement (le doc_id vient du registre, jamais d'un utilisateur)."""
        with self._conn() as c:
            return self._d(c.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone())

    def get(self, tenant_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        security.check_doc_id(doc_id)
        with self._conn() as c:
            return self._d(c.execute("SELECT * FROM documents WHERE doc_id = ? AND tenant_id = ?",
                                     (doc_id, tenant_id)).fetchone())

    def find_duplicate(self, tenant_id: str, owner_id: str, content_hash: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            return self._d(c.execute(
                "SELECT * FROM documents WHERE tenant_id=? AND owner_id=? AND content_hash=? "
                "AND status IN ('pending','processing','indexed') LIMIT 1",
                (tenant_id, owner_id, content_hash)).fetchone())

    def find_previous_version(self, tenant_id: str, owner_id: str, source_name: str, exclude_doc_id: str):
        with self._conn() as c:
            return self._d(c.execute(
                "SELECT * FROM documents WHERE tenant_id=? AND owner_id=? AND source_name=? "
                "AND status='indexed' AND doc_id != ? ORDER BY version DESC LIMIT 1",
                (tenant_id, owner_id, source_name, exclude_doc_id)).fetchone())

    def list_tenant(self, tenant_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM documents WHERE tenant_id=? AND status NOT IN ('superseded','deleted') "
                             "ORDER BY created_at DESC", (tenant_id,)).fetchall()
        return [self._d(r) for r in rows]


def can_see(doc: Dict[str, Any], user_id: str, role: str, admin_roles) -> bool:
    """Même logique de visibilité que la recherche vectorielle (+ les admins voient tout leur tenant)."""
    if role in admin_roles or doc["owner_id"] == user_id:
        return True
    if doc["visibility"] == "tenant":
        return True
    return doc["visibility"] == "roles" and role in doc["allowed_roles"]
