import json
import re
from openai import OpenAI
from config import LLM_MODEL_NAME, OLLAMA_BASE_URL, SEMANTIC_CURATED_DIR
import yaml
import unicodedata
import requests

def _info(el: dict) -> dict:
    meta = el.get("meta") if isinstance(el.get("meta"), dict) else {}
    syn = meta.get("synonymes", [])
    return {
        "title": el.get("title") or "",
        "description": el.get("description") or meta.get("description", "") or "",
        "synonymes": syn if isinstance(syn, list) else [],
    }


def extract_allowed_members() -> dict:
    """Retourne {cube: {"measures": {nom: info}, "dimensions": {...}, "time": {...}, "joins": [...]}}"""
    cubes_idx = {}
    if not SEMANTIC_CURATED_DIR.exists():
        print(f"❌ Dossier sémantique introuvable : {SEMANTIC_CURATED_DIR}")
        return cubes_idx

    for file_path in SEMANTIC_CURATED_DIR.glob("**/*.yml"):
        try:
            data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"⚠️ Lecture impossible {file_path.name} : {e}")
            continue
        for cube in data.get("cubes", []) or []:
            c = cube.get("name")
            if not c:
                continue
            entry = {"measures": {}, "dimensions": {}, "time": {},
                     "joins": [j.get("name") for j in (cube.get("joins") or []) if j.get("name")]}
            for m in cube.get("measures", []) or []:
                if m.get("name"):
                    entry["measures"][f"{c}.{m['name']}"] = _info(m)
            for d in cube.get("dimensions", []) or []:
                if d.get("name"):
                    bucket = "time" if d.get("type") == "time" else "dimensions"   # type uniquement, pas le nom
                    entry[bucket][f"{c}.{d['name']}"] = _info(d)
            cubes_idx[c] = entry

    print(f"✅ {len(cubes_idx)} cube(s) chargé(s)")
    return cubes_idx

VALID_GRANULARITY = ["day", "week", "month", "quarter", "year"]
VALID_RANGES = ["today", "yesterday", "this week", "last week", "this month", "last month",
                "this quarter", "last quarter", "this year", "last year",
                "last 7 days", "last 30 days"]

SYSTEM_PROMPT = """You translate ONE dashboard component into ONE Cube.js query.
Return ONLY a JSON object that follows the provided schema.
Rules:
- Use only the members listed below, copied exactly (format cube.member). Never add suffixes like .sum or .count.
- The aggregation is already inside each measure: do not add any.
- Choose the measure whose title/description best matches the component meaning.
- Do not invent filters or values.
"""


def _tokens(text: str) -> set:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return {t[:5] for t in re.split(r"[^a-z0-9]+", text) if len(t) > 2}   # racine à 5 lettres: stock/stocks


def _score(comp_tokens: set, name: str, info: dict) -> int:
    hay = _tokens(" ".join([name.replace("_", " "), info["title"], info["description"], " ".join(info["synonymes"])]))
    return len(comp_tokens & hay)


def reachable(cubes_idx: dict, root: str) -> set:
    """Cubes reliés à `root` par des jointures (graphe non orienté, approximation)."""
    adj = {c: set() for c in cubes_idx}
    for c, e in cubes_idx.items():
        for j in e["joins"]:
            if j in adj:
                adj[c].add(j)
                adj[j].add(c)
    seen, stack = {root}, [root]
    while stack:
        for n in adj.get(stack.pop(), ()):
            if n not in seen:
                seen.add(n)
                stack.append(n)
    return seen


def shortlist_members(comp: dict, cubes_idx: dict, n_meas=8, n_dim=12):
    ct = _tokens(f"{comp.get('title', '')} {comp.get('description', '')}")

    def top(kind, cubes, n):
        items = [(m, i) for c in cubes for m, i in cubes_idx[c][kind].items()]
        items.sort(key=lambda x: _score(ct, *x), reverse=True)
        return items[:n]

    all_meas = top("measures", list(cubes_idx), 1)
    if not all_meas:
        return [], [], []
    root = all_meas[0][0].split(".")[0]
    pool = reachable(cubes_idx, root)           # sans jointure: uniquement le cube racine
    return top("measures", pool, n_meas), top("dimensions", pool, n_dim), top("time", pool, 4)


def build_schema(meas: list, dims: list, times: list) -> dict:
    props = {
        "measures": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string", "enum": meas}},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        "order_direction": {"type": "string", "enum": ["asc", "desc"]},
    }
    if dims:
        props["dimensions"] = {"type": "array", "maxItems": 2, "items": {"type": "string", "enum": dims}}
    if times:
        props["timeDimensions"] = {"type": "array", "maxItems": 1, "items": {
            "type": "object",
            "properties": {
                "dimension": {"type": "string", "enum": times},
                "granularity": {"type": "string", "enum": VALID_GRANULARITY},
                "dateRange": {"type": "string", "enum": VALID_RANGES},
            },
            "required": ["dimension"],
        }}
    return {"type": "object", "properties": props, "required": ["measures"]}


def normalize_query(q: dict, ctype: str, time_names: set) -> dict:
    """Règles déterministes selon le type de composant (le LLM ne décide pas de la forme)."""
    measures = list(dict.fromkeys(q.get("measures", [])))
    dims = [d for d in q.get("dimensions", []) if d not in time_names]
    tds = [t for t in q.get("timeDimensions", []) if t.get("dimension") in time_names]
    limit = q.get("limit")
    ctype = (ctype or "").lower()

    if "metric" in ctype:
        measures, dims, limit = measures[:1], [], None
        tds = [{"dimension": t["dimension"], **({"dateRange": t["dateRange"]} if t.get("dateRange") else {})}
               for t in tds[:1]]
    elif "line" in ctype or "area" in ctype:
        measures, dims = measures[:1], []
        tds = [{**t, "granularity": t.get("granularity") or "month"} for t in tds[:1]]
        limit = None
    elif "pie" in ctype or "donut" in ctype:
        measures, dims, tds, limit = measures[:1], dims[:1], [], limit or 8
    elif "bar" in ctype:
        measures, dims, tds, limit = measures[:1], dims[:1], [], limit or 15
    elif "table" in ctype:
        dims, limit = dims[:2], limit or 20

    query = {"measures": measures}
    if dims:
        query["dimensions"] = dims
    if tds:
        query["timeDimensions"] = tds
    if limit:
        query["limit"] = limit
        query["order"] = {measures[0]: q.get("order_direction", "desc")}
    return query


def validate_query(query: dict, cubes_idx: dict):
    flat = {}
    for cname, c in cubes_idx.items():
        for kind in ("measures", "dimensions", "time"):
            for m in c[kind]:
                flat[m] = (kind, cname)

    errors = []
    if not query.get("measures"):
        errors.append("no measure")
    for m in query.get("measures", []):
        if flat.get(m, (None,))[0] != "measures":
            errors.append(f"unknown measure {m}")
    for d in query.get("dimensions", []):
        if flat.get(d, (None,))[0] != "dimensions":
            errors.append(f"{d} is not a regular dimension")
    for t in query.get("timeDimensions", []):
        if flat.get(t.get("dimension"), (None,))[0] != "time":
            errors.append(f"{t.get('dimension')} is not a time dimension")

    if not errors:
        used = {flat[m][1] for m in query["measures"] + query.get("dimensions", [])
                + [t["dimension"] for t in query.get("timeDimensions", [])]}
        root = flat[query["measures"][0]][1]
        missing = used - reachable(cubes_idx, root)
        if missing:
            errors.append(f"no join path from {root} to {sorted(missing)}")
    return (not errors), "; ".join(errors)


def cube_dry_run(query: dict):
    """True/False = Cube accepte/refuse. None = Cube injoignable (on ne bloque pas)."""
    headers = {"Authorization": CUBE_API_TOKEN} if CUBE_API_TOKEN else {}
    try:
        r = requests.get(f"{CUBE_API_URL}/v1/sql", params={"query": json.dumps(query)},
                         headers=headers, timeout=20)
        if r.status_code == 200:
            return True, ""
        try:
            return False, r.json().get("error", r.text)
        except Exception:
            return False, r.text
    except Exception as e:
        return None, str(e)


def _fmt(items: list) -> str:
    lines = []
    for n, i in items:
        syn = f" (syn: {', '.join(i['synonymes'])})" if i["synonymes"] else ""
        lines.append(f"- {n} | {i['title']} {i['description']}{syn}".rstrip())
    return "\n".join(lines) or "(none)"


def generate_one_query(client, comp: dict, cubes_idx: dict, model: str, max_attempts: int = 3) -> dict:
    title, ctype = comp.get("title", ""), comp.get("type", "")
    meas, dims, times = shortlist_members(comp, cubes_idx)
    if not meas:
        return {"component_title": title, "cube_query": None, "status": "failed", "error": "no measure available"}

    time_names = {n for n, _ in times}
    schema = build_schema([n for n, _ in meas], [n for n, _ in dims], [n for n, _ in times])
    system = (f"{SYSTEM_PROMPT}\n### Measures\n{_fmt(meas)}\n### Dimensions\n{_fmt(dims)}\n"
              f"### Time dimensions\n{_fmt(times)}")

    feedback, last_error = "", ""
    for attempt in range(1, max_attempts + 1):
        user = json.dumps(comp, ensure_ascii=False) + feedback
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "cube_query", "strict": True, "schema": schema}},
                temperature=0.0,
            )
            raw = json.loads(resp.choices[0].message.content)
        except Exception as e:
            last_error = f"LLM/JSON error: {e}"
            feedback = f"\nPrevious answer was invalid JSON. Fix it."
            print(f"⚠️ [{title}] tentative {attempt} : {last_error}")
            continue

        query = normalize_query(raw, ctype, time_names)
        ok, err = validate_query(query, cubes_idx)
        status = "ok"
        if ok:
            print(f"the query is validate",query, status )
            """
            dry_ok, dry_err = cube_dry_run(query)
            if dry_ok is False:
                ok, err = False, dry_err
            elif dry_ok is None:
                status = "unchecked"      # Cube injoignable
            """
        if ok:
            print(f"✅ [{title}] tentative {attempt} ({status})")
            return {"component_title": title, "component_type": ctype, "cube_query": query, "status": status}

        last_error = err
        feedback = f"\nYour previous answer {json.dumps(query)} was rejected: {err}. Fix it."
        print(f"⚠️ [{title}] tentative {attempt} rejetée : {err}")

    return {"component_title": title, "component_type": ctype, "cube_query": None,
            "status": "failed", "error": last_error}



def generate_cube_queries_for_dashboard(dashboard_json: dict) -> list[dict]:
    """
    Génère un tableau de requêtes Cube.js (une pour CHAQUE composant du JSON LanceDB)
    avec validation stricte des membres via qwen2.5-coder:7b et logs complets.
    """
    print("\n--- [CUBE QUERY GENERATOR] Début de la génération ---")
    
    components = dashboard_json.get("components", [])
    if not components:
        print("⚠️ [AVERTISSEMENT] Aucun composant trouvé dans le JSON du dashboard.")
        return []

    print(f"📊 [DEBUG] Nombre de composants à traduire : {len(components)}")

    cubes_idx = extract_allowed_members()
    if not any(c["measures"] for c in cubes_idx.values()):
        print("❌ Aucune mesure trouvée dans les YAML curés.")
        return []

    client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
    coder_model = "qwen2.5-coder:7b"

    # Un appel par composant: contexte court = moins d'hallucinations
    results = [generate_one_query(client, comp, cubes_idx, coder_model) for comp in components]

    ok = sum(1 for r in results if r["cube_query"])
    print(f"✅ {ok}/{len(results)} requête(s) valide(s)")
    print("--- [CUBE QUERY GENERATOR] Fin du traitement ---\n")
    return results