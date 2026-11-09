import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Set

import lancedb
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.state import SMAState

log = logging.getLogger(__name__)
settings = get_settings()

TABLE_NAME = "erp_enterprises"

SYSTEM_ANSWER_PREFIXES = (
    "Je ne dispose pas de suffisamment",
    "Désolé, aucune documentation",
    "Erreur",
)


# --- Modèle Pydantic pour le résultat d'évaluation ---
class SecurityEvaluationResult(BaseModel):
    is_valid: bool = Field(
        description="True si la réponse respecte les permissions RBAC de l'entreprise."
    )
    reason: str = Field(
        description="Explication explicite en cas de violation des règles d'accès."
    )


def _get_lancedb_path() -> str:
    """Retourne le chemin d'accès à la base LanceDB sous forme de chaîne de caractères."""
    if hasattr(settings, "lancedb_uri") and settings.lancedb_uri:
        return str(settings.lancedb_uri)
    if hasattr(settings, "data_dir") and settings.data_dir:
        return str(Path(settings.data_dir) / "lancedb")
    return "data/lancedb"


# --- Fonctions d'extraction RBAC dans LanceDB (Zero-Vector / Full Arrow) ---
def get_allowed_screens_from_lancedb(tenant_id: str, user_role: str) -> Set[str]:
    """
    Récupère la liste des écrans autorisés pour un rôle et un tenant directement depuis LanceDB.
    """
    db = lancedb.connect(_get_lancedb_path())
    try:
        table = db.open_table(TABLE_NAME)
    except Exception:
        return set()

    role_norm = user_role.lower().strip()
    escaped_tenant = tenant_id.replace("'", "''")

    flt = (
        f"tenant_id = '{escaped_tenant}' AND "
        f"doc_type = 'rbac_role' AND "
        f"(allowed_roles LIKE '%|{role_norm}|%' OR allowed_roles LIKE '%|all|%' OR allowed_roles LIKE '%|*|%')"
    )

    try:
        results = table.search().where(flt, prefilter=True).to_list()
        allowed_screens = set()
        for r in results:
            screens = [s.strip() for s in r.get("text", "").split(",") if s.strip()]
            allowed_screens.update(screens)
        return allowed_screens
    except Exception as e:
        log.error(f"[EVALUATOR] Erreur de lecture des écrans autorisés LanceDB : {e}")
        return set()


def get_all_tenant_screens_from_lancedb(tenant_id: str) -> Set[str]:
    """
    Récupère l'ensemble de TOUS les écrans enregistrés pour une entreprise.
    """
    db = lancedb.connect(_get_lancedb_path())
    try:
        table = db.open_table(TABLE_NAME)
        escaped_tenant = tenant_id.replace("'", "''")
        results = (
            table.search()
            .where(
                f"tenant_id = '{escaped_tenant}' AND doc_type = 'rbac_role'",
                prefilter=True,
            )
            .to_list()
        )
        all_screens = set()
        for r in results:
            screens = [s.strip() for s in r.get("text", "").split(",") if s.strip()]
            all_screens.update(screens)
        return all_screens
    except Exception as e:
        log.error(f"[EVALUATOR] Erreur de lecture de tous les écrans LanceDB : {e}")
        return set()


def find_forbidden_screens(answer: str, user_role: str, tenant_id: str) -> List[str]:
    """
    Détecte de manière ultra-rapide si la réponse mentionne un écran interdit au rôle.
    """
    allowed = get_allowed_screens_from_lancedb(tenant_id, user_role)
    all_screens = get_all_tenant_screens_from_lancedb(tenant_id)

    if not all_screens:
        # Aucun paramétrage RBAC trouvé pour cette entreprise : par sécurité, on laisse passer
        return []

    # Les écrans strictement interdits pour cet utilisateur
    forbidden = all_screens - allowed
    if not forbidden:
        return []

    found = []
    answer_lower = answer.lower()

    for screen in forbidden:
        screen_clean = screen.strip().lower()
        if not screen_clean:
            continue
        # Utilisation de regex boundary pour éviter de fausses détections sur des sous-mots
        pattern = r"\b" + re.escape(screen_clean) + r"\b"
        if re.search(pattern, answer_lower):
            found.append(screen)

    return found


# --- Nœud du Graphe LangGraph ---
def evaluator_judge_node(state: SMAState) -> Dict[str, Any]:
    """
    Nœud Évaluateur RBAC Déterministe :
    1. Filtre les réponses système / échecs RAG (pas besoin d'évaluation).
    2. Récupère le tenant_id et le user_role depuis jwt_claims.
    3. Effectue la vérification déterministe via LanceDB en < 1ms.
    4. Réoriente vers la régénération si violation détectée (max 3 retries).
    """
    answer = state.get("answer", "")
    jwt_claims = state.get("jwt_claims", {}) or {}

    tenant_id = jwt_claims.get("tenant_id", "default")
    user_role = jwt_claims.get("role", "employe")
    retry = state.get("retry_count", 0)

    # 1. Ignorer les messages système ou d'absence de documentation
    if any(
        answer.startswith(prefix) for prefix in SYSTEM_ANSWER_PREFIXES
    ) or not state.get("retrieved_docs"):
        return {
            "evaluation_passed": True,
            "evaluation_feedback": "Réponse système ou hors périmètre, non soumise au contrôle RBAC.",
        }

    # 2. Contrôle RBAC déterministe via LanceDB
    forbidden = find_forbidden_screens(answer, user_role, tenant_id)
    reason = ""
    if forbidden:
        reason = (
            f"La réponse mentionne des écrans ou modules non autorisés pour le rôle '{user_role}' "
            f"dans l'entreprise '{tenant_id}' : {', '.join(forbidden)}."
        )

    # 3. Validation réussie
    if not reason:
        return {
            "evaluation_passed": True,
            "evaluation_feedback": "Réponse conforme aux permissions du rôle.",
        }

    # 4. Échec et gestion de la boucle de rétroaction
    retry += 1
    if retry >= 3:
        log.warning(
            f"[EVALUATOR] Blocage définitif de la réponse après 3 tentatives pour {tenant_id}/{user_role}"
        )
        return {
            "evaluation_passed": False,
            "retry_count": retry,
            "evaluation_feedback": reason,
            "answer": (
                f"Je ne peux pas vous guider sur cette demande avec votre profil actuel ({user_role}). "
                "Certaines fonctionnalités mentionnées relèvent d'autres habilitations au sein de votre entreprise. "
                "Veuillez contacter votre administrateur."
            ),
        }

    log.info(f"[EVALUATOR] Rejet de la réponse (essai {retry}/3). Raison : {reason}")
    return {
        "evaluation_passed": False,
        "retry_count": retry,
        "evaluation_feedback": reason,
    }
