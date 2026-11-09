import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.ingestion import security

GLOBAL_TENANT = "global"            # documentation officielle partagée par toutes les entreprises
VISIBILITIES = ("tenant", "roles", "private")


@dataclass
class Block:
    """Unité sémantique extraite d'un document (page, diapositive, groupe de lignes...)."""
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)   # page, slide, row_start, row_end, section_title...


@dataclass
class ExtractedDocument:
    blocks: List[Block]
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    extraction_method: str = "text_layer"
    confidence: float = 1.0
    partial: bool = False            # True si une partie du contenu n'a pas pu être extraite


@dataclass
class IngestContext:
    """Identité et droits de l'envoi. tenant_id / owner_id / role viennent TOUJOURS du JWT."""
    tenant_id: str
    owner_id: str
    role: str
    visibility: str = "private"      # valeur par défaut la plus restrictive
    allowed_roles: List[str] = field(default_factory=list)
    doc_type: str = "document"
    module: str = ""
    tags: List[str] = field(default_factory=list)

    def validate(self) -> "IngestContext":
        self.tenant_id = security.check_tenant(self.tenant_id)
        self.owner_id = security.check_user(self.owner_id)
        self.role = security.check_role(self.role)
        if self.visibility not in VISIBILITIES:
            raise ValueError(f"visibility doit être l'une de {VISIBILITIES}")
        self.allowed_roles = sorted({security.check_role(r) for r in self.allowed_roles})
        if self.visibility == "roles" and not self.allowed_roles:
            raise ValueError("visibility='roles' exige au moins un rôle dans allowed_roles")
        if self.visibility != "roles":
            self.allowed_roles = []
        self.doc_type = security.check_slug(self.doc_type or "document", "doc_type")
        self.module = re.sub(r"[\x00-\x1f]", "", (self.module or "").strip())[:80]
        self.tags = [security.check_slug(t, "tag") for t in self.tags if t and t.strip()][:10]
        return self
