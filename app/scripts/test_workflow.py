"""Test du workflow SMA : 1) retrieval seul, 2) workflow LangGraph complet.

Placement : le même dossier que ton test actuel (la racine du projet est ajoutée au sys.path).
Usage :
    python test_workflow.py                    # tous les scénarios
    python test_workflow.py --retrieval-only   # uniquement embedding + LanceDB (sans LLM)
    python test_workflow.py --only 1           # un seul scénario (numéro)
"""

import sys
import time
import traceback
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.ingestion.store import vector_store
from app.graph.workflow import sma_graph

settings = get_settings()

# (libellé, question, rôle, écran courant, comportement attendu)
SCENARIOS = [
    (
        "Financier : créer une facture fournisseur",
        "Comment puis-je créer une facture fournisseur ?",
        "financier",
        "Factures Fournisseurs",
        "ANSWER",
    ),
    (
        "RH : approuver un congé",
        "Comment approuver une demande de congé ?",
        "rh",
        "Gestion des Congés",
        "ANSWER",
    ),
    (
        "Employé : changer son mot de passe (écran commun)",
        "Comment changer mon mot de passe ?",
        "all",
        "Accueil",
        "ANSWER",
    ),
    (
        "RBAC : RH demande une fonction Finance",
        "Comment créer une facture fournisseur ?",
        "rh",
        "Gestion des Congés",
        "REFUSAL",
    ),
    (
        "RBAC : employé demande la gestion des utilisateurs",
        "Comment créer un nouvel utilisateur et lui attribuer un rôle ?",
        "all",
        "Accueil",
        "REFUSAL",
    ),
    (
        "Hors sujet",
        "Quelle est la capitale de la France ?",
        "financier",
        "Accueil",
        "REFUSAL",
    ),
]


def embed(text: str):
    r = requests.post(
        f"{settings.ollama_base_url}/api/embeddings",
        json={"model": settings.ollama_embed_model, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def check_retrieval(query: str, role: str) -> bool:
    """Étape 1 : isole le retrieval (embedding + filtre RBAC LanceDB), sans LLM."""
    print("  [1] Retrieval seul")
    try:
        vec = embed(query)
    except Exception as e:
        print(f"      ❌ Embedding impossible ({type(e).__name__}: {e})")
        return False
    print(f"      embedding OK (dimension {len(vec)})")

    docs = vector_store.search_guide(query_vector=vec, user_role=role, top_k=10)
    print(f"      {len(docs)} documents récupérés pour le rôle '{role}'")
    leaks = [d for d in docs if d.get("role_autorise") not in (role, "all")]
    for d in docs[:5]:
        print(
            f"        - {d.get('ecran'):<26} rôle={d.get('role_autorise'):<10} dist={d.get('_distance', 0):.3f}"
        )
    if leaks:
        print(f"      ❌ FUITE RBAC : {len(leaks)} document(s) hors rôle !")
        return False
    if not docs:
        print(
            "      ⚠️  Aucun document : table vide, rôle inconnu ou base non re-seedée ?"
        )
    return bool(docs)


def classify(answer: str) -> str:
    """Classe la réponse finale : ERROR (panne technique), REFUSAL ou ANSWER."""
    a = (answer or "").strip()
    if a.startswith("Erreur"):
        return "ERROR"
    if a.startswith(
        ("Je ne dispose pas", "Désolé, aucune", "Je ne peux pas vous guider")
    ):
        return "REFUSAL"
    return "ANSWER"


def run_workflow(query: str, role: str, page: str) -> dict:
    """Étape 2 : workflow LangGraph complet."""
    print("  [2] Workflow complet")
    state = {
        "raw_query": query,
        "jwt_claims": {"role": role, "sub": "user_123"},
        "current_page": page,
        "retry_count": 0,
    }
    t0 = time.time()
    result = sma_graph.invoke(state)
    print(f"      Durée                : {time.time() - t0:.1f} s")
    docs = result.get("retrieved_docs") or []
    print(f"      Évaluation validée   : {result.get('evaluation_passed')}")
    print(f"      Tentatives           : {result.get('retry_count')}")
    print(f"      Feedback évaluateur  : {result.get('evaluation_feedback')}")
    print(
        f"      Docs après reranking : {[(d.get('ecran'), round(d.get('rerank_score', 0), 3)) for d in docs]}"
    )
    print(f"      Sources              : {result.get('sources')}")
    print(f"      Réponse finale       :\n{indent(result.get('answer', ''))}")
    return result


def indent(text: str, n: int = 8) -> str:
    return "\n".join(" " * n + line for line in str(text).splitlines())


def main():
    args = sys.argv[1:]
    retrieval_only = "--retrieval-only" in args
    only = int(args[args.index("--only") + 1]) if "--only" in args else None

    print("🚀 Test du workflow LangGraph\n")
    summary = []
    for i, (label, query, role, page, expected) in enumerate(SCENARIOS, 1):
        if only and i != only:
            continue
        print("=" * 80)
        print(
            f"#{i} {label}\n  question : {query}\n  rôle     : {role} | écran : {page} | attendu : {expected}"
        )
        status = "?"
        try:
            retrieval_ok = check_retrieval(query, role)
            if retrieval_only:
                status = "retrieval OK" if retrieval_ok else "retrieval KO"
            else:
                res = run_workflow(query, role, page)
                kind = classify(res.get("answer", ""))
                status = "OK" if kind == expected else f"ÉCART ({kind})"
        except Exception:
            traceback.print_exc()
            status = "ERREUR"
        summary.append((i, label, expected, status))

    print("\n" + "=" * 80 + "\nRÉSUMÉ")
    for i, label, expected, status in summary:
        print(f"  #{i} {status:<16} (attendu {expected:<7}) {label}")


if __name__ == "__main__":
    main()
