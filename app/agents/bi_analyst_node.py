# Fichier : app/agents/bi_analyst_node.py
import json
import logging
import os
import re
from datetime import date, timedelta
from typing import Any, Dict

import httpx
import lancedb

from app.core.artefacts_generation import determine_recommended_chart
from app.core.cube_client import execute_cube_query
from app.core.db_manager import get_latest_connection_config
from app.graph.state import SMAState
from app.ingestion.settings import get_cfg

logger = logging.getLogger(__name__)

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
LLM_MODEL_NAME = os.getenv("BI_LLM_MODEL", "qwen2.5-coder:7b")
LLM_SUMMARY_NAME = os.getenv("OLLAMA_LLM_GUIDE_ERP_MODEL", "qwen2.5:7b")


def get_tenant_semantic_context(tenant_id: str) -> tuple[list[dict], str]:
    """Récupère à la fois les modèles Cube YAML et le glossaire depuis tenant_semantic_models."""
    print(f"\n🔍 [DEBUG] Recherche du contexte sémantique pour le tenant : {tenant_id}")
    try:
        cfg = get_cfg()
        db_path = getattr(cfg, "lancedb_uri", cfg.data_dir / "lancedb")
        db = lancedb.connect(str(db_path))

        if "tenant_semantic_models" not in db.table_names():
            print(
                "⚠️ [DEBUG] La table 'tenant_semantic_models' n'existe pas dans LanceDB."
            )
            return [], ""

        table = db.open_table("tenant_semantic_models")
        df = table.to_pandas()

        tenant_df = df[df["tenant_id"] == tenant_id.lower()]
        if tenant_df.empty:
            print(
                f"⚠️ [DEBUG] Aucun enregistrement trouvé dans LanceDB pour le tenant '{tenant_id}'."
            )
            return [], ""

        models = []
        glossary_parts = []

        for _, row in tenant_df.iterrows():
            table_name = row.get("table_name", "")
            content = row.get("yaml_content", "")

            if table_name.startswith("_glossary_"):
                print(f"📖 [DEBUG] Glossaire détecté : {table_name}")
                glossary_parts.append(content)
            else:
                print(f"📊 [DEBUG] Modèle Cube YAML détecté : {table_name}")
                print(f"📊 [DEBUG] Cube détecté : {table_name}")
                models.append({"table_name": table_name, "yaml_content": content})

        print(
            f"✅ [DEBUG] Contexte chargé : {len(models)} modèle(s) YAML et {len(glossary_parts)} partie(s) de glossaire."
        )
        return models, "\n\n".join(glossary_parts)
    except Exception as e:
        print(
            f"❌ [DEBUG ERROR] Erreur lors de la lecture de tenant_semantic_models : {e}"
        )
        logger.warning(f"⚠️ Erreur lors de la lecture de tenant_semantic_models : {e}")
        return [], ""


# ------------------------- SYSTEM PROMPT ----------------------------
CUBE_SYSTEM_PROMPT = """You are a translation engine: "natural-language question -> Cube.js query (REST API /v1/load)".
You are NOT a conversational assistant. Your output is parsed by a program, never read by a human.
A single syntax error breaks the whole pipeline. Be absolutely rigorous.

## TIME CONTEXT (source of truth, NEVER guess a date)
- Today's date: <<TODAY>>
- Current month: <<CURRENT_MONTH>>
- Previous month: <<PREVIOUS_MONTH>>
- Current year: <<CURRENT_YEAR>>
- Previous year: <<PREVIOUS_YEAR>>

## ALLOWED MEMBERS (closed list, copy them character by character)
### Measures (go in "measures")
<<MEASURES>>
### Dimensions (go in "dimensions", or can be used in "filters")
<<DIMENSIONS>>
### Time dimensions (go in "timeDimensions")
<<TIME_DIMENSIONS>>

<<GLOSSARY>>

## OUTPUT FORMAT
Reply with ONE single JSON object: no markdown, no ``` fences, no comments, no text before or after.
Allowed keys (all optional except "measures"):
{
  "measures":       ["cube.measure", ...],
  "dimensions":     ["cube.dimension", ...],
  "filters":        [{"member": "cube.dimension", "operator": "equals", "values": ["value"]}],
  "timeDimensions": [{"dimension": "cube.date", "dateRange": "last month", "granularity": "month"}],
  "order":          {"cube.measure": "desc"},
  "limit":          10
}

## STRICT RULES
1. "measures" and "dimensions" are arrays of STRINGS of the form "cube_name.member_name". NEVER objects like {"name": ..., "type": ...}.
2. Use ONLY members from the lists above. Never invent a name, never translate it, never change its case.
3. The user's question may be in French (or another language). Use the business glossary and the member descriptions to map the user's terms to the right members.
4. A filter ALWAYS contains "member", "operator" and "values" (except "set"/"notSet": no "values").
   "values" is ALWAYS an array of strings, even for numbers: ["2026"], not [2026].
   Allowed operators: equals, notEquals, contains, notContains, startsWith, endsWith, gt, gte, lt, lte, set, notSet, inDateRange, beforeDate, afterDate.
5. FORBIDDEN: placeholders, variables, templates, functions or expressions ({{...}}, ${...}, NOW(), "previous", etc.).
   Every value must be CONCRETE. Use the time context above.
   Example: "last month" -> the value "<<PREVIOUS_MONTH>>" when the dimension is a text month in YYYY-MM format.
6. Dates : si un champ temporel (`timeDimensions`) est utilisé, le `dateRange` doit OBLIGATOIREMENT utiliser les mots-clés natifs de Cube.js (ex: `"last month"`, `"this year"`, `"last year"`) ou un tableau de dates explicites `["YYYY-MM-DD", "YYYY-MM-DD"]`. 
   INTERDIT d'utiliser des expressions comme `"previous month"` (toujours utiliser `"last month"`).
   Si le mois/année est une colonne textuelle au format YYYY-MM (ex: `rmonth`), utilise plutôt un filtre `"equals"` avec la variable `<<PREVIOUS_MONTH>>`.
7. "per X", "by X", "for each X", "grouped by X" -> X goes in "dimensions". "total", "sum", "how many" -> measure alone, no dimension.
8. "top N", "best", "highest" -> "order" (desc) + "limit": N. "lowest", "worst" -> "order" asc.
9. "List", "Give me the list", "Donne-moi la liste" -> Put the entity name/ID in "dimensions" and include a generic count measure (e.g., `cube.count`) because Cube.js requires at least one measure. Use "order" to sort results cleanly.
10. Only use members from a single cube, unless the models explicitly show a join between cubes.
11. If NO measure in the list matches the question, return exactly {"measures": []}.
12. No trailing commas, double quotes only, strictly valid JSON.

## EXAMPLES (member names below are FICTIONAL: use only the allowed list above)
Question: "Total quantity per warehouse last month"
{"measures":["stock.quantity"],"dimensions":["stock.warehouse_name"],"filters":[{"member":"stock.month","operator":"equals","values":["<<PREVIOUS_MONTH>>"]}]}

Question: "Top 5 products by revenue this year"
{"measures":["sales.revenue"],"dimensions":["sales.product_name"],"timeDimensions":[{"dimension":"sales.order_date","dateRange":"this year"}],"order":{"sales.revenue":"desc"},"limit":5}

Question: "Number of orders whose customer contains Dupont"
{"measures":["sales.count"],"filters":[{"member":"sales.customer_name","operator":"contains","values":["Dupont"]}]}

Question: "Donne-moi la liste de nos clients par région"
{"measures":["clients.count"],"dimensions":["clients.region","clients.name"],"order":{"clients.region":"asc","clients.name":"asc"}}

## INCORRECT EXAMPLE (NEVER reproduce this)
{"measures":[{"name":"stock.quantity","type":"sum"}],"dimensions":[{"name":"stock.warehouse_name","type":"string"}],"filters":[{"member":"stock.month","operator":"equals","values":["{{date.month.previous}}"]}]}
Errors: objects instead of strings; placeholder instead of a concrete value.
"""


# -------------------- FONCTION D'EXTRACTION DES DONNEES GENERES PAR LE MODELE QWEN ---------------------
class CubeQueryError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


VALID_OPERATORS = [
    "equals",
    "notEquals",
    "contains",
    "notContains",
    "startsWith",
    "endsWith",
    "gt",
    "gte",
    "lt",
    "lte",
    "set",
    "notSet",
    "inDateRange",
    "beforeDate",
    "afterDate",
]
VALID_GRANULARITIES = [
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "quarter",
    "year",
]

_SECTION_RE = {
    "measures": re.compile(
        r"^-\s*Mesures[^:]*:\s*(.*?)(?=\n-\s|\Z)", re.I | re.M | re.S
    ),
    "dimensions": re.compile(
        r"^-\s*Dimensions[^:]*:\s*(.*?)(?=\n-\s|\Z)", re.I | re.M | re.S
    ),
}
_ITEM_RE = re.compile(r"(\w+)\s*(?:\((\w+)\))?")  # "nom" ou "nom (type)"


def extract_cube_members(semantic_models: list[dict]) -> dict[str, dict[str, str]]:
    """Parse le résumé texte d'un cube -> {measures, dimensions, time_dimensions}."""
    members = {"measures": {}, "dimensions": {}, "time_dimensions": {}}
    for model in semantic_models:
        cube = model["table_name"]
        text = model.get("yaml_content") or ""
        for kind, pattern in _SECTION_RE.items():
            match = pattern.search(text)
            if not match:
                continue
            for raw_item in re.split(r"[,\n]+", match.group(1)):
                item = _ITEM_RE.fullmatch(raw_item.strip())
                if not item:
                    continue
                name, typ = item.group(1), (item.group(2) or "").lower()
                key = kind
                if kind == "dimensions" and (
                    typ == "time" or (not typ and "date" in name.lower())
                ):
                    key = "time_dimensions"
                members[key][f"{cube}.{name}"] = ""
    return members


def _render(items: dict[str, str]) -> str:
    return (
        "\n".join(f"- {k}  ({v})" if v else f"- {k}" for k, v in items.items())
        or "- (aucun)"
    )


def build_cube_system_prompt(members: dict, glossary_context: str) -> str:
    today = date.today()
    prev_last_day = today.replace(day=1) - timedelta(days=1)
    glossary = (
        f"## GLOSSAIRE MÉTIER (sens des tables et colonnes)\n{glossary_context}"
        if glossary_context
        else ""
    )
    replacements = {
        "<<TODAY>>": today.isoformat(),
        "<<CURRENT_MONTH>>": today.strftime("%Y-%m"),
        "<<PREVIOUS_MONTH>>": prev_last_day.strftime("%Y-%m"),
        "<<CURRENT_YEAR>>": str(today.year),
        "<<PREVIOUS_YEAR>>": str(today.year - 1),
        "<<MEASURES>>": _render(members["measures"]),
        "<<DIMENSIONS>>": _render(members["dimensions"]),
        "<<TIME_DIMENSIONS>>": _render(members["time_dimensions"]),
        "<<GLOSSARY>>": glossary,
    }
    prompt = CUBE_SYSTEM_PROMPT
    for token, value in replacements.items():
        prompt = prompt.replace(token, value)
    return prompt


def build_cube_json_schema(members: dict) -> dict:
    measures = list(members["measures"])
    times = list(members["time_dimensions"])
    all_dims = list(members["dimensions"]) + times
    str_enum = lambda values: (
        {"type": "string", "enum": values} if values else {"type": "string"}
    )

    return {
        "type": "object",
        "properties": {
            "measures": {"type": "array", "items": str_enum(measures)},
            "dimensions": {"type": "array", "items": str_enum(all_dims)},
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "member": str_enum(all_dims + measures),
                        "operator": {"type": "string", "enum": VALID_OPERATORS},
                        "values": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["member", "operator"],
                },
            },
            "timeDimensions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dimension": str_enum(times),
                        "dateRange": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ]
                        },
                        "granularity": {"type": "string", "enum": VALID_GRANULARITIES},
                    },
                    "required": ["dimension"],
                },
            },
            "order": {"type": "object"},
            "limit": {"type": "integer"},
        },
        "required": ["measures"],
    }


def _as_member(item: Any) -> str:
    """Tolère un objet {"name": ...} produit par erreur et le ramène à une chaîne."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(
            item.get("name") or item.get("member") or item.get("dimension") or ""
        ).strip()
    return ""


def normalize_and_validate_cube_query(raw: dict, members: dict) -> dict:
    errors: list[str] = []
    ok_measures = set(members["measures"])
    ok_times = set(members["time_dimensions"])
    ok_dims = set(members["dimensions"]) | ok_times
    query: dict[str, Any] = {}

    measures = [m for m in (_as_member(x) for x in raw.get("measures") or []) if m]
    if not measures:
        errors.append("Aucune mesure ne correspond à la question.")
    unknown = [m for m in measures if m not in ok_measures]
    if unknown:
        errors.append(
            f"Mesures inconnues {unknown}. Autorisées : {sorted(ok_measures)[:40]}"
        )
    query["measures"] = measures

    dimensions = [d for d in (_as_member(x) for x in raw.get("dimensions") or []) if d]
    unknown = [d for d in dimensions if d not in ok_dims]
    if unknown:
        errors.append(
            f"Dimensions inconnues {unknown}. Autorisées : {sorted(ok_dims)[:40]}"
        )
    if dimensions:
        query["dimensions"] = dimensions

    filters = []
    for f in raw.get("filters") or []:
        if not isinstance(f, dict):
            errors.append(f"Filtre invalide : {f}")
            continue
        member, op = _as_member(f.get("member")), f.get("operator")
        values = [str(v) for v in (f.get("values") or [])]
        if member not in ok_dims | ok_measures:
            errors.append(f"Filtre : membre inconnu '{member}'.")
        if op not in VALID_OPERATORS:
            errors.append(f"Filtre : opérateur invalide '{op}'.")
        if op not in ("set", "notSet") and not values:
            errors.append(f"Filtre sur '{member}' : 'values' vide.")
        if any("{{" in v or "${" in v for v in values):
            errors.append(
                f"Filtre sur '{member}' : placeholder interdit {values}, utilise une valeur concrète."
            )
        item = {"member": member, "operator": op}
        if op not in ("set", "notSet"):
            item["values"] = values
        filters.append(item)
    if filters:
        query["filters"] = filters

    tds = []
    for td in raw.get("timeDimensions") or []:
        dim = _as_member(td.get("dimension")) if isinstance(td, dict) else ""
        if dim not in ok_times:
            errors.append(
                f"timeDimension inconnue '{dim}'. Autorisées : {sorted(ok_times)}"
            )
            continue
        item = {"dimension": dim}
        if td.get("dateRange"):
            if "{{" in str(td["dateRange"]):
                errors.append("dateRange : placeholder interdit.")
            item["dateRange"] = td["dateRange"]
        if td.get("granularity") in VALID_GRANULARITIES:
            item["granularity"] = td["granularity"]
        tds.append(item)
    if tds:
        query["timeDimensions"] = tds

    order = raw.get("order")
    if isinstance(order, dict) and order:
        clean = {
            k: v
            for k, v in order.items()
            if v in ("asc", "desc") and k in ok_dims | ok_measures
        }
        if clean:
            query["order"] = clean
    if isinstance(raw.get("limit"), int) and raw["limit"] > 0:
        query["limit"] = raw["limit"]

    if errors:
        raise CubeQueryError(errors)
    return query


async def generate_cube_query_with_qwen(
    user_query: str, semantic_models: list[dict[str, str]], glossary_context: str
) -> Dict[str, Any]:
    members = extract_cube_members(semantic_models)
    if not members["measures"]:
        raise ValueError("Aucune mesure exploitable dans les cubes indexés du tenant.")

    system_prompt = build_cube_system_prompt(members, glossary_context)
    schema = build_cube_json_schema(members)

    last_errors: list[str] = []
    last_raw = ""
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        prompt = f'Question de l\'utilisateur : "{user_query}"\n\nJSON :'
        if last_errors:
            prompt = (
                f'Question de l\'utilisateur : "{user_query}"\n\n'
                f"Ta réponse précédente était invalide :\n{last_raw}\n"
                "Erreurs à corriger :\n- "
                + "\n- ".join(last_errors)
                + "\n\nJSON corrigé :"
            )

        payload = {
            "model": LLM_MODEL_NAME,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "format": schema,  # sortie contrainte par schéma JSON
            "options": {"temperature": 0.0, "num_ctx": 8192},
        }

        async with httpx.AsyncClient(timeout=3600.0) as client:
            response = await client.post(OLLAMA_API_URL, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Erreur d'appel à Qwen (Ollama) : {response.text}")

        last_raw = response.json().get("response", "{}")
        print(f"\n--- 📥 [QWEN tentative {attempt}] ---\n{last_raw}\n")

        try:
            cleaned = last_raw.replace("```json", "").replace("```", "").strip()
            return normalize_and_validate_cube_query(json.loads(cleaned), members)
        except json.JSONDecodeError as e:
            last_errors = [f"JSON invalide : {e}"]
        except CubeQueryError as e:
            last_errors = e.errors
        print(f"⚠️ [CUBE VALIDATION] Tentative {attempt} rejetée : {last_errors}")

    raise RuntimeError(
        "Impossible de générer une requête Cube valide : " + "; ".join(last_errors)
    )


async def bi_analyst_node(state: SMAState) -> Dict[str, Any]:
    """
    Nœud LangGraph pour l'intention 'bi_analyst' via Cube, Qwen et le glossaire LanceDB.
    """
    # ----------------------- on vérifie d'abord si la variable sql_queries figure déja dans le cache => l'utilisateur à déja
    existing_sql_query = state.get("sql_result")

    query = state.get("reformulated_query") or state.get("raw_query", "")
    jwt_claims = state.get("jwt_claims", {}) or {}

    tenant_id = jwt_claims.get("tenant_id")
    user_id = jwt_claims.get("sub", state.get("user_id", "user_001"))

    print(
        f"\n📊 [BI ANALYST NODE] ➔ Tenant: {tenant_id} | User: {user_id} | Query: {query}"
    )
    logger.info(
        f"📊 [BI ANALYST NODE] Tenant: {tenant_id} | User: {user_id} | Query: {query}"
    )

    if not query or not query.strip():
        print("⚠️ [BI ANALYST] Question vide.")
        return {
            "error": "Aucune question n'a été formulée pour l'analyse BI.",
            "answer": "Veuillez formuler une requête analytique valide.",
            "route_completed": True,
        }

    if not tenant_id:
        print("⚠️ [BI ANALYST] Tenant_id absent.")
        return {
            "error": "Identifiant d’entreprise absent des claims.",
            "answer": "Impossible d’identifier l’entreprise pour cette analyse.",
            "route_completed": True,
        }

    try:
        if not existing_sql_query:
            # Récupération des modèles YAML et du glossaire depuis LanceDB
            semantic_models, glossary_context = get_tenant_semantic_context(tenant_id)
            if not semantic_models:
                print("❌ [BI ANALYST] Aucun modèle sémantique trouvé.")
                return {
                    "error": "Aucun modèle sémantique n’est indexé pour ce tenant.",
                    "answer": (
                        "L’administrateur doit charger les modèles YAML dans la couche "
                        "sémantique LanceDB avant de lancer une analyse."
                    ),
                    "route_completed": True,
                }

            db_config = get_latest_connection_config(tenant_id)
            if not db_config:
                print(
                    "❌ [BI ANALYST] Aucune configuration de base de données trouvée."
                )
                return {
                    "error": "Aucune connexion analytique n’est configurée.",
                    "answer": (
                        "L’administrateur doit configurer la base analytique depuis son "
                        "profil avant de lancer une analyse."
                    ),
                    "route_completed": True,
                }

            # 1. Le LLM (Qwen) traduit la question en requête JSON Cube en combinant YAML + Glossaire
            print("🤖 [QWEN] Traduction de la question en cours par le LLM...")
            cube_query = await generate_cube_query_with_qwen(
                query, semantic_models, glossary_context
            )
            print(
                f"📋 [CUBE QUERY GÉNÉRÉE] :\n{json.dumps(cube_query, indent=2, ensure_ascii=False)}"
            )
            logger.info(f"📋 [CUBE QUERY GÉNÉRÉE] {cube_query}")

            # 2. Appel de Cube avec le contexte de sécurité du tenant et la requête générée
            print(
                f"🚀 [CUBE CLIENT] Envoi de la requête à Cube.js pour le tenant '{tenant_id}'..."
            )

        # Si la requête est stockée en string JSON dans l'état, on la recharge en dict, sinon on la prend telle quelle
        if existing_sql_query:
            cube_query = (
                json.loads(existing_sql_query)
                if isinstance(existing_sql_query, str)
                else existing_sql_query
            )

        cube_result = await execute_cube_query(
            tenant_id=tenant_id, cube_query=cube_query, db_config=db_config
        )

        data = cube_result.get("data", [])
        print(f"✅ [BI SUCCESS] {len(data)} ligne(s) récupérée(s) de Cube.js.")
        print(f"✅ [BI SUCCESS] {data} données récupérées depuis cube.")
        print(json.dumps(cube_query, ensure_ascii=False))

        logger.info(
            f"✅ [BI SUCCESS] {len(data)} lignes récupérées pour le tenant {tenant_id}."
        )

        summary = f"Analyse BI réalisée avec succès. {len(data)} résultat(s) trouvé(s)."

        # 3. Appel au modèle pour reformuler les résultats bruts en langage naturel
        natural_answer_prompt = f"""
            Tu es un assistant de Business Intelligence expert et bienveillant.
            L'utilisateur a posé la question suivante : "{query}"

            Voici les données brutes récupérées de la base de données analytique (Cube.js) :
            {json.dumps(data, ensure_ascii=False, indent=2)}

            Rédige une réponse et des métadonnées de visualisation sous forme d'un objet JSON strict avec les clés suivantes :
            - "answer": Rédige une réponse claire en français pour l'utilisateur, Si le résultat contient plusieurs lignes (tableaux, classements), renvoie-les proprement sous forme de texte ou de liste à puces, Ne mentionne pas de détails techniques (pas de JSON, pas de noms de tables techniques comme 'rv_st_stockmovement').
            - "title": un titre court et pertinent pour un graphique représentant ces données.
            - "xAxis": le libellé de l'axe horizontal (ou de la dimension principale).
            - "yAxis": le libellé de l'axe vertical (ou de la mesure principale).

            Réponds UNIQUEMENT avec un objet JSON valide, sans texte additionnel, sans balises markdown.
        """

        payload = {
            "model": LLM_SUMMARY_NAME,
            "prompt": natural_answer_prompt,
            "format": "json",  # Force Ollama à renvoyer du JSON strict
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 8192},
        }

        async with httpx.AsyncClient(timeout=3600.0) as client:
            response = await client.post(OLLAMA_API_URL, json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"Erreur d'appel à Qwen (Ollama) : {response.text}")

        # Parsing de la réponse JSON structurée du LLM
        response_json = json.loads(response.json().get("response", "{}"))

        print("response_json", response_json)

        resume = response_json.get("answer", "Aucun résultat textuel généré.")
        chart_title = response_json.get("title", "Analyse des données")
        x_axis_label = response_json.get("xAxis", "Dimension")
        y_axis_label = response_json.get("yAxis", "Valeur")

        print("response_json", resume, chart_title, x_axis_label, y_axis_label)

        # Détermination automatique du type de graphique via l'algorithme déterministe
        chart_recommendation = determine_recommended_chart(cube_query, data)

        print("chart recommandation", chart_recommendation)

        # Construction du dictionnaire de configuration du graphique pour le Front-end
        chart_config = {
            "title": chart_title,
            "type": chart_recommendation["type"],
            "reason": chart_recommendation["reason"],
            "xAxisLabel": x_axis_label,
            "yAxisLabel": y_axis_label,
            "measures": cube_query.get("measures", []),
            "dimensions": cube_query.get("dimensions", []),
            "timeDimensions": [
                td["dimension"] for td in cube_query.get("timeDimensions", [])
            ],
        }

        return {
            "sql_query": json.dumps(cube_query, ensure_ascii=False),
            "sql_result": {"data": data},
            "answer": resume,
            "chart_config": chart_config,  # Transmis au Front-end
            "route_completed": True,
        }

    except Exception as e:
        print(f"❌ [BI ANALYST ERROR] {type(e).__name__}: {e}")
        logger.error(f"❌ [BI ANALYST ERROR] {type(e).__name__}: {e}")
        return {
            "error": str(e),
            "answer": f"Une erreur technique est survenue lors de l'analyse : {str(e)}",
            "route_completed": False,
        }
