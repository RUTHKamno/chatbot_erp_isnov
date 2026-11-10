import streamlit as st
import json

# Imports corrigés (sans le préfixe "src.")
from nlp_processor import process_user_query
from vector_retriever import retrieve_business_context
from json_generator import generate_dashboard_json
from cube_query_generator import generate_cube_queries_for_dashboard
from dashboard_view import render_dashboard
from semantic_manager import (
    sync_cube_models_to_curated, 
    get_all_curated_models, 
    load_yaml_model, 
    save_yaml_model
)
import yaml
import re
import copy
from streamlit_sortables import sort_items
from cube_executor import run_cube_buffer

# Gestion de l'import de la configuration selon son emplacement :
# Si config.py est à la racine (dashboard_generator/config.py) :
import sys
from pathlib import Path
# Ajoute le dossier parent au chemin de recherche Python si nécessaire
sys.path.append(str(Path(__file__).parent.parent))
#from config import LLM_MODEL_CODER_NAME, EMBEDDING_MODEL_NAME, CUBE_SOURCE_MODEL_DIR

# Si config.py est directement dans src/ :
from config import LLM_MODEL_CODER_NAME, EMBEDDING_MODEL_NAME, CUBE_SOURCE_MODEL_DIR

# Configuration de la page Streamlit
st.set_page_config(
    page_title="ERP Intelligence & Dashboard Generator",
    page_icon="📊",
    layout="wide"
)

st.title("🤖 Plateforme d'Intelligence Décisionnelle (RAG + Cube Semantic Layer)")

# --- NAVIGATION PAR ONGLETS ---
tab1, tab2 = st.tabs(["📊 Générateur de Dashboards", "📚 Éditeur de Couche Sémantique (Cube)"])

# =========================================================================
# ONGLSET 1 : GÉNÉRATEUR DE DASHBOARDS
# =========================================================================
with tab1:
    

    st.header("Générateur Dynamique de Dashboards par Thématique")
    st.markdown("Pose ta question ou décris ton besoin métier, et l'IA générera la structure de ton tableau de bord en s'appuyant sur les règles métier.")

    user_query = st.text_input(
        "Quel type de dashboard souhaitez-vous générer ?",
        placeholder="Ex: Je veux un dashboard pour analyser l'état de mes stocks...",
        key="dashboard_query_input"
    )

    if "last_dashboard_json" in st.session_state:
        st.markdown("---")

        if st.checkbox("🛠️ Mode Debug (Afficher les flux techniques & JSON)", value=False, key="dash_debug"):
            debug_tabs = st.tabs(
                ["NLP", "LanceDB", "JSON dashboard", "Requêtes Cube", "Exécution Cube"]
            )
            with debug_tabs[0]:
                st.json(st.session_state.get("last_nlp_result", {}))
            with debug_tabs[1]:
                lancedb_results = st.session_state.get("last_lancedb_results", [])
                if lancedb_results:
                    st.json(lancedb_results)
                else:
                    st.info("LanceDB n’a retourné aucun contexte.")
                st.caption("Contexte effectivement transmis au générateur JSON :")
                st.json(st.session_state.get("last_contexts_used", []))
            with debug_tabs[2]:
                st.json(st.session_state["last_dashboard_json"])
            with debug_tabs[3]:
                st.json(st.session_state.get("last_cube_queries", []))
            with debug_tabs[4]:
                st.json(st.session_state.get("last_buffered_data", []))

        render_dashboard(st.session_state["last_buffered_data"], st.session_state["last_dashboard_json"])

    

    if st.button("🚀 Générer la structure du Dashboard", type="primary"):
        if not user_query.strip():
            st.warning("⚠️ Veuillez entrer une question ou un besoin valide.")
        else:
            with st.status("🔍 Analyse de la question et recherche dans la base...", expanded=True) as status:
                st.write("Étape 1 : Nettoyage NLP et détection de l'intention...")
                nlp_result = process_user_query(user_query)
                domain = nlp_result["domain_intent"]
                st.write(f"-> Domaine détecté : **`{domain}`**")

                st.write("Étape 2 : Recherche des règles métier dans LanceDB...")
                retrieved_contexts = retrieve_business_context(
                    cleaned_query=nlp_result["cleaned_query"],
                    domain_intent=domain,
                    limit=3
                )
                lancedb_results = retrieved_contexts
                contexts_used = retrieved_contexts or [
                    "Aucun contexte spécifique trouvé. Utilise les standards généraux."
                ]

                st.write(f"Étape 3 : Génération du JSON via {LLM_MODEL_CODER_NAME}...")
                dashboard_json = generate_dashboard_json(user_query, contexts_used)
                # --- ÉTAPE 4 : Génération des requêtes Cube.js via qwen2.5-coder:7b ---
                st.write("Étape 4 : Génération des requêtes Cube.js via qwen2.5-coder:7b...")
                cube_queries = generate_cube_queries_for_dashboard(dashboard_json)

                # --- ÉTAPE 5 : Exécution et Buffer d'accumulation ---
                st.write("Étape 5 : Exécution des requêtes sur Cube.js et remplissage du buffer...")
                buffered_dashboard_data = run_cube_buffer(cube_queries)

                # On stocke les résultats dans la session Streamlit pour les figer à l'affichage
                st.session_state["last_dashboard_json"] = dashboard_json
                st.session_state["last_cube_queries"] = cube_queries
                st.session_state["last_buffered_data"] = buffered_dashboard_data
                st.session_state["last_nlp_result"] = nlp_result
                st.session_state["last_lancedb_results"] = lancedb_results
                st.session_state["last_contexts_used"] = contexts_used
                st.session_state["last_domain"] = domain
                st.session_state["dash_title"] = dashboard_json.get("dashboard_title") or "Tableau de bord"
                st.session_state["dash_run"] = st.session_state.get("dash_run", 0) + 1   # relance l'animation
                
                status.update(label="✅ Génération et exécution terminées !", state="complete", expanded=False)
                
                #status.update(label="✅ Génération terminée !", state="complete", expanded=False)

            st.rerun()


# =========================================================================
# ONGLET 2 : ÉDITEUR SÉMANTIQUE (style Helical / Human-in-the-Loop)
# =========================================================================
MEASURE_TYPES = ["count", "count_distinct", "count_distinct_approx", "sum",
                 "avg", "min", "max", "number", "running_total"]
DIM_TYPES = ["string", "number", "time", "boolean", "geo"]
MEASURE_FORMATS = ["", "percent", "currency"]
DIM_FORMATS = ["", "id", "link", "imageUrl"]
NAME_RE = re.compile(r"^[_a-zA-Z][_a-zA-Z0-9]*$")


def get_work_model(file):
    """Copie de travail en session : les modifs survivent aux reruns Streamlit."""
    store = st.session_state.setdefault("work_models", {})
    if file not in store:
        model = copy.deepcopy(load_yaml_model(file)) or {}
        for c in model.get("cubes", []):
            c["dimensions"] = c.get("dimensions") or []
            c["measures"] = c.get("measures") or []
            # migration : meta.description -> description (standard Cube)
            for el in c["dimensions"] + c["measures"]:
                el["name"] = str(el.get("name", ""))
                meta = el.get("meta")
                if isinstance(meta, dict) and meta.get("description") and not el.get("description"):
                    el["description"] = meta.pop("description")
                elif isinstance(meta, dict) and "description" in meta:
                    meta.pop("description")
        store[file] = model
    return store[file]


def measure_to_dimension(m):
    d = {k: v for k, v in m.items()
         if k not in ("type", "format", "filters", "rolling_window", "drill_members")}
    d["type"] = "number"
    d["sql"] = m.get("sql") or m["name"]
    return d


def dimension_to_measure(d):
    m = {k: v for k, v in d.items()
         if k not in ("type", "format", "primary_key", "case", "sub_query", "granularity")}
    m["type"] = "sum" if d.get("type") == "number" else "count_distinct"
    m["sql"] = d.get("sql") or d["name"]
    return m


def validate_cube(cube):
    errors, warnings = [], []
    dims, meas = cube.get("dimensions", []), cube.get("measures", [])
    names = [e.get("name") for e in dims + meas]
    for n in set(names):
        if names.count(n) > 1:
            errors.append(f"Nom en double dans le cube : `{n}`")
    for kind, items, types in (("Dimension", dims, DIM_TYPES), ("Mesure", meas, MEASURE_TYPES)):
        for e in items:
            n = e.get("name", "?")
            if not NAME_RE.match(str(n)):
                errors.append(f"{kind} `{n}` : nom invalide (lettres, chiffres, _ ; pas de chiffre au début)")
            if e.get("type") not in types:
                errors.append(f"{kind} `{n}` : type `{e.get('type')}` invalide pour une {kind.lower()}")
            if not e.get("sql") and not (kind == "Mesure" and e.get("type") == "count"):
                errors.append(f"{kind} `{n}` : `sql` manquant")
            meta = e.get("meta") or {}
            if not (e.get("title") or e.get("description") or meta.get("synonymes")):
                warnings.append(f"{kind} `{n}` : aucun titre / description / synonyme pour l'IA")
    return errors, warnings


def _set(obj, key, value):
    """Écrit la valeur, ou retire la clé si vide (YAML propre)."""
    if value in ("", None, []):
        obj.pop(key, None)
    else:
        obj[key] = value


with tab2:
    st.header("⚙️ Curation Sémantique & Enrichissement IA")
    st.caption("Glisse les éléments entre Dimensions et Mesures pour corriger le classement, "
               "sélectionne un élément pour l'enrichir (formule, synonymes, contexte IA), "
               "et contrôle le YAML en direct.")

    ss = st.session_state
    ss.setdefault("dnd_v", 0)   # force le rechargement du composant drag & drop
    ss.setdefault("rst", 0)     # force le reset des champs après annulation

    # Sélection en attente (rename / ajout / suppression) : doit être appliquée AVANT le selectbox
    if "_pending_sel" in ss:
        ss["sel_elem"] = ss.pop("_pending_sel")

    curated_models = get_all_curated_models()

    top1, top2, top3 = st.columns([1.2, 3, 2])
    with top1:
        st.write("")
        if st.button("🔄 Synchroniser depuis Cube", key="sync_cube_btn"):
            if sync_cube_models_to_curated():
                ss.pop("work_models", None)
                ss["dnd_v"] += 1
                st.toast("Synchronisation réussie !")
                st.rerun()

    if not curated_models:
        st.info("Aucun fichier sémantique trouvé. Vérifie le dossier source Cube et clique sur Synchroniser.")
    else:
        with top2:
            selected_model_file = st.selectbox("📂 Modèle sémantique", curated_models, key="select_curated_model")

        model = get_work_model(selected_model_file)
        cubes_list = model.get("cubes", [])

        if not cubes_list:
            st.warning("Aucun bloc 'cubes' détecté dans ce fichier YAML.")
        else:
            with top3:
                cube_name = st.selectbox("🧊 Cube", [c.get("name") for c in cubes_list], key="select_cube")
            cube = next(c for c in cubes_list if c.get("name") == cube_name)
            dims, meas = cube["dimensions"], cube["measures"]
            dim_names = [d["name"] for d in dims]
            mea_names = [m["name"] for m in meas]
            all_names = dim_names + mea_names

            if ss.get("sel_elem") not in all_names:
                ss["sel_elem"] = all_names[0] if all_names else None

            st.markdown("---")
            col_dnd, col_edit, col_prev = st.columns([1, 1.2, 1.3], gap="large")

            # ------------------------------------------------------------------
            # COLONNE 1 : Glisser-déposer Dimensions <-> Mesures
            # ------------------------------------------------------------------
            with col_dnd:
                st.subheader("🗂️ Classement")
                st.caption("Glisse un élément d'une liste à l'autre pour le convertir.")

                PH_D, PH_M = "— aucune dimension —", "— aucune mesure —"
                st.caption(f"{len(dim_names)} dimension(s) · {len(mea_names)} mesure(s)")

                if len(set(all_names)) != len(all_names):
                    st.error("Des noms sont en double : corrige-les avant d'utiliser le glisser-déposer.")
                else:
                    containers = [
                        {"header": "🏷️ Dimensions", "items": dim_names or [PH_D]},
                        {"header": "📏 Mesures", "items": mea_names or [PH_M]},
                    ]
                    safe_key = re.sub(r"\W", "_", f"dnd_{selected_model_file}_{cube_name}")
                    result = sort_items(
                        containers,
                        multi_containers=True,
                        direction="vertical",
                        key=f"{safe_key}_{ss['dnd_v']}_{ss['rst']}",
                    )
                    new_dims = [n for n in result[0]["items"] if n not in (PH_D, PH_M)]
                    new_meas = [n for n in result[1]["items"] if n not in (PH_D, PH_M)]

                    # Garde-fou : on n'applique que si on retrouve exactement les mêmes éléments
                    same_items = sorted(new_dims + new_meas) == sorted(all_names)
                    if same_items and (new_dims != dim_names or new_meas != mea_names):
                        dim_by = {d["name"]: d for d in dims}
                        mea_by = {m["name"]: m for m in meas}
                        cube["dimensions"] = [dim_by[n] if n in dim_by else measure_to_dimension(mea_by[n])
                                              for n in new_dims]
                        cube["measures"] = [mea_by[n] if n in mea_by else dimension_to_measure(dim_by[n])
                                            for n in new_meas]
                        ss["dnd_v"] += 1
                        st.rerun()
                with st.popover("➕ Ajouter un élément", use_container_width=True):
                    add_name = st.text_input("Nom technique", key="add_name")
                    add_kind = st.radio("Type", ["Dimension", "Mesure"], horizontal=True, key="add_kind")
                    if st.button("Créer", key="add_btn"):
                        if not NAME_RE.match(add_name) or add_name in all_names:
                            st.error("Nom invalide ou déjà utilisé.")
                        else:
                            if add_kind == "Dimension":
                                cube["dimensions"].append({"name": add_name, "sql": add_name, "type": "string"})
                            else:
                                cube["measures"].append({"name": add_name, "type": "count"})
                            ss["_pending_sel"] = add_name
                            ss["dnd_v"] += 1
                            st.rerun()

            # ------------------------------------------------------------------
            # COLONNE 2 : Panneau d'édition (équivalent du clic droit Helical)
            # ------------------------------------------------------------------
            with col_edit:
                st.subheader("🛠️ Édition")

                if not all_names:
                    st.info("Ce cube ne contient aucun élément.")
                else:
                    def _icon(n):
                        return ("📏 " if n in mea_names else "🏷️ ") + n

                    sel = st.selectbox("Élément à éditer (tape pour rechercher)", all_names,
                                       format_func=_icon, key="sel_elem")
                    kind = "measure" if sel in mea_names else "dimension"
                    obj = next(e for e in (meas if kind == "measure" else dims) if e["name"] == sel)
                    kp = f"{ss['rst']}|{selected_model_file}|{cube_name}|{sel}|{kind}"

                    # --- Général
                    with st.expander("Général", expanded=True):
                        new_name = st.text_input("Nom technique", value=obj["name"], key=f"{kp}|name")
                        if new_name != obj["name"]:
                            if NAME_RE.match(new_name) and new_name not in all_names:
                                obj["name"] = new_name
                                ss["_pending_sel"] = new_name
                                ss["dnd_v"] += 1
                                st.rerun()
                            else:
                                st.error("Nom invalide ou déjà utilisé.")
                        _set(obj, "title", st.text_input("Titre affiché", value=obj.get("title", ""), key=f"{kp}|title"))
                        _set(obj, "description", st.text_area("Description", value=obj.get("description", ""),
                                                              key=f"{kp}|desc", height=80))

                    # --- Formule / type
                    with st.expander("Formule & Type", expanded=True):
                        types = MEASURE_TYPES if kind == "measure" else DIM_TYPES
                        cur = obj.get("type")
                        options = types if cur in types else ([cur] + types if cur else types)
                        label = "Agrégation (`type`)" if kind == "measure" else "Type sémantique (`type`)"
                        obj["type"] = st.selectbox(label, options, index=options.index(cur) if cur in options else 0,
                                                   key=f"{kp}|type")
                        if obj["type"] not in types:
                            st.warning("Type non valide pour ce genre d'élément : choisis-en un autre.")
                        _set(obj, "sql", st.text_area("Formule / colonne SQL (`sql`)", value=obj.get("sql", ""),
                                                      key=f"{kp}|sql", height=80,
                                                      help="Ex : quantite * prix_unitaire, ou {autre_mesure} / {CUBE}.col"))
                        fmts = MEASURE_FORMATS if kind == "measure" else DIM_FORMATS
                        curf = obj.get("format", "")
                        fopts = fmts if curf in fmts else fmts + [curf]
                        _set(obj, "format", st.selectbox("Format", fopts, index=fopts.index(curf), key=f"{kp}|fmt"))

                    # --- Contexte IA
                    with st.expander("🤖 Contexte IA", expanded=True):
                        meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
                        cur_syn = meta.get("synonymes", [])
                        cur_syn = cur_syn if isinstance(cur_syn, list) else []
                        cur_ex = meta.get("exemples", [])
                        cur_ex = cur_ex if isinstance(cur_ex, list) else []

                        syn = st.text_input("Synonymes (séparés par des virgules)", value=", ".join(cur_syn),
                                            key=f"{kp}|syn")
                        instr = st.text_area("Instructions pour l'IA", value=meta.get("instructions", ""),
                                             key=f"{kp}|instr", height=80,
                                             placeholder="Ex : à utiliser pour toute question sur la valeur du stock")
                        ex = st.text_area("Exemples de questions (une par ligne)", value="\n".join(cur_ex),
                                          key=f"{kp}|ex", height=80)

                        _set(meta, "synonymes", [s.strip() for s in syn.split(",") if s.strip()])
                        _set(meta, "instructions", instr)
                        _set(meta, "exemples", [l.strip() for l in ex.splitlines() if l.strip()])
                        _set(obj, "meta", meta)

                    if st.button("🗑️ Supprimer cet élément", key=f"{kp}|del"):
                        (meas if kind == "measure" else dims).remove(obj)
                        ss["_pending_sel"] = None
                        ss["dnd_v"] += 1
                        st.rerun()
                    target = "dimension" if kind == "measure" else "mesure"
                    if st.button(f"🔁 Convertir en {target}", key=f"{kp}|conv"):
                        if kind == "measure":
                            meas.remove(obj)
                            dims.append(measure_to_dimension(obj))
                        else:
                            dims.remove(obj)
                            meas.append(dimension_to_measure(obj))
                        ss["_pending_sel"] = obj["name"]
                        ss["dnd_v"] += 1
                        st.rerun()

            # ------------------------------------------------------------------
            # COLONNE 3 : Aperçu + validation + sauvegarde
            # ------------------------------------------------------------------
            with col_prev:
                st.subheader("👁️ Aperçu & Validation")
                errors, warnings = validate_cube(cube)

                if errors:
                    st.error(f"{len(errors)} erreur(s) bloquante(s) :\n\n" + "\n".join(f"- {e}" for e in errors))
                else:
                    st.success("✅ Le cube est valide pour Cube.")
                if warnings:
                    with st.expander(f"⚠️ {len(warnings)} élément(s) sans contexte IA"):
                        st.markdown("\n".join(f"- {w}" for w in warnings))

                tab_full, tab_elem = st.tabs(["YAML complet", "Élément sélectionné"])
                with tab_full:
                    st.code(yaml.dump(model, allow_unicode=True, sort_keys=False), language="yaml")
                with tab_elem:
                    if all_names and sel in all_names:
                        st.code(yaml.dump(obj, allow_unicode=True, sort_keys=False), language="yaml")

                st.markdown("---")
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("💾 Valider et enregistrer", type="primary",
                                 disabled=bool(errors), use_container_width=True, key="save_yaml_btn"):
                        try:
                            save_yaml_model(selected_model_file, model)
                            st.success(f"`{selected_model_file}` enregistré avec succès !")
                        except Exception as e:
                            st.error(f"Erreur lors de l'enregistrement : {e}")
                with b2:
                    if st.button("↩️ Annuler les modifications", use_container_width=True, key="reset_yaml_btn"):
                        ss["work_models"].pop(selected_model_file, None)
                        ss["rst"] += 1
                        ss["dnd_v"] += 1
                        st.rerun()