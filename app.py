# Fichier : app_streamlit.py
import hashlib
import html
import io
import json
import logging
import os
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from sqlalchemy.exc import SQLAlchemyError
from streamlit_mermaid import st_mermaid

from app.authentication import (
    authenticate_user,
    create_access_token,
    create_tenant_user,
    list_tenant_users,
)
from app.core.audio_processor import clean_and_save_audio

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_WHITE = ASSETS_DIR / "logo_isnov_blanc.png"
LOGO_COLOR = ASSETS_DIR / "logo_isnov.png"
FAVICON = ASSETS_DIR / "favicon.png"

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")


# Palette sobre : ardoise + vert ISNOV (accent unique)
SLATE_900 = "#1F2A37"
SLATE_600 = "#52606D"
SLATE_200 = "#D9DEE4"
SLATE_100 = "#E8ECF0"
BG = "#F5F7F9"
ACCENT = "#5F8F37"
ACCENT_DARK = "#4A7329"

st.set_page_config(
    page_title="ISNOV | Assistant ERP",
    page_icon=Image.open(FAVICON) if FAVICON.exists() else None,
    layout="wide",
)
st.session_state.setdefault("current_detected_lang", "fr")


# L'executor est créé UNE seule fois (et non à chaque rerun du script).
@st.cache_resource
def get_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2)


SCRAPE_EXECUTOR = get_executor()

logger = logging.getLogger(__name__)


# ==============================================================================
# STYLE
# ==============================================================================
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, .stApp, [data-testid="stSidebar"] {{
    font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
}}
.stApp {{ background: {BG}; }}

/* Chrome Streamlit discret */
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stDeployButton"] {{ display: none; }}
[data-testid="stHeader"] {{ background: transparent; }}

.block-container {{ max-width: 1080px; padding-top: 2.4rem; padding-bottom: 3rem; }}

/* Barre latérale */
[data-testid="stSidebar"] {{ background: {SLATE_900}; border-right: none; }}
[data-testid="stSidebar"] * {{ color: #DDE3EA; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.12); }}
.sidebar-caption {{ font-size: .78rem; color: #9AA5B1 !important; line-height: 1.5; }}
.sidebar-section {{ font-size: .85rem; font-weight: 600; color: #FFFFFF !important; margin: 1.2rem 0 .3rem; }}

/* En-tête de page */
.page-title {{ font-size: 1.65rem; font-weight: 600; color: {SLATE_900}; margin: 0; }}
.page-sub {{ font-size: .95rem; color: {SLATE_600}; margin: .35rem 0 0; max-width: 70ch; }}
.page-rule {{ width: 44px; height: 3px; background: {ACCENT}; border-radius: 2px; margin: 1rem 0 1.6rem; }}

/* Boutons */
.stButton > button, .stDownloadButton > button {{
    border-radius: 6px; border: 1px solid #C5CCD3; background: #FFFFFF;
    color: {SLATE_900}; font-weight: 500; padding: .5rem 1.1rem; box-shadow: none;
}}
.stButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT_DARK}; background: #FFFFFF; }}
.stButton > button:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; }}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background: {ACCENT}; border-color: {ACCENT}; color: #FFFFFF;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background: {ACCENT_DARK}; border-color: {ACCENT_DARK}; color: #FFFFFF;
}}

/* Cartes et conteneurs */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #FFFFFF; border-color: {SLATE_100}; border-radius: 8px;
}}
[data-testid="stChatMessage"] {{
    background: #FFFFFF; border: 1px solid {SLATE_100}; border-radius: 8px; padding: .9rem 1rem;
}}
[data-testid="stAlert"] {{ border-radius: 8px; }}
[data-testid="stFileUploaderDropzone"] {{
    background: #FFFFFF; border: 1px dashed #B8C1CA; border-radius: 8px;
}}
[data-testid="stTextInput"] input {{ border-radius: 6px; }}
h2, h3 {{ color: {SLATE_900}; font-weight: 600; }}

/* Impression du dashboard */
@media print {{
    [data-testid="stSidebar"], [data-testid="stHeader"],
    [data-testid="stToolbar"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [class*="st-key-dash_toolbar"], [class*="st-key-dash_actions_"] {{
        display: none !important;
    }}
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main {{
        height: auto !important; overflow: visible !important; position: static !important;
    }}
    .stApp {{ background: #FFFFFF !important; }}
    .block-container {{ max-width: 100% !important; padding: 0 !important; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        break-inside: avoid; page-break-inside: avoid;
    }}
    * {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='page-title'>{title}</div>"
        f"<div class='page-sub'>{subtitle}</div>"
        f"<div class='page-rule'></div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# DASHBOARDS : dashboards charges uniquement dans le state
# ==============================================================================
# ==============================================================================
# DASHBOARD (POC : stocké dans la session)
# ==============================================================================
def _table_snapshot(df: pd.DataFrame) -> dict:
    return {
        "kind": "table",
        "table_json": df.to_json(orient="split", date_format="iso"),
    }


def dashboard_items() -> list[dict]:
    return st.session_state.setdefault("dashboard_items", [])


def dashboard_add(snapshot: dict, title: str) -> bool:
    items = dashboard_items()
    payload = json.dumps(snapshot, sort_keys=True, default=str)
    item_id = hashlib.md5(payload.encode()).hexdigest()[:8]
    if any(i["id"] == item_id for i in items):
        return False  # déjà présent (même contenu)
    items.append(
        {
            "id": item_id,
            "title": title,
            "added_at": datetime.now().isoformat(timespec="seconds"),
            **snapshot,
        }
    )
    return True


def dashboard_remove(item_id: str) -> None:
    st.session_state.dashboard_items = [
        i for i in dashboard_items() if i["id"] != item_id
    ]


def dashboard_move(item_id: str, delta: int) -> None:
    items = dashboard_items()
    idx = next((n for n, i in enumerate(items) if i["id"] == item_id), None)
    if idx is None or not 0 <= idx + delta < len(items):
        return
    items[idx], items[idx + delta] = items[idx + delta], items[idx]


def dashboard_rename(item_id: str) -> None:
    new_title = st.session_state.get(f"dash_title_{item_id}", "").strip()
    if new_title:
        for i in dashboard_items():
            if i["id"] == item_id:
                i["title"] = new_title


def dashboard_clear() -> None:
    st.session_state.dashboard_items = []


def render_snapshot(item: dict, key: str) -> None:
    kind = item["kind"]
    if kind == "figure":
        fig = pio.from_json(item["fig_json"])
        fig.update_layout(title=None, margin=dict(t=10))  # le titre est dans la carte
        st.plotly_chart(fig, use_container_width=True, key=f"{key}_fig")
    elif kind == "table":
        df = pd.read_json(io.StringIO(item["table_json"]), orient="split")
        st.table(df.head(20))  # st.table = HTML statique, s'imprime correctement
    elif kind == "kpi":
        for col, m in zip(st.columns(len(item["metrics"])), item["metrics"]):
            col.metric(m["label"], m["value"])
    if item.get("caption"):
        st.caption(item["caption"])


def build_dashboard_html(items: list[dict], title: str, subtitle: str) -> str:
    """Export HTML autonome (ouvrir puis Ctrl+P). Plotly.js chargé via CDN."""
    cards, js_loaded = [], False
    for item in items:
        if item["kind"] == "figure":
            fig = pio.from_json(item["fig_json"])
            fig.update_layout(title=None, margin=dict(t=10))
            body = fig.to_html(
                full_html=False, include_plotlyjs=False if js_loaded else "cdn"
            )
            js_loaded = True
        elif item["kind"] == "table":
            df = pd.read_json(io.StringIO(item["table_json"]), orient="split")
            body = df.head(20).to_html(index=False, border=0)
        else:
            body = "".join(
                f"<div class='kpi'><small>{html.escape(m['label'])}</small>"
                f"<b>{html.escape(m['value'])}</b></div>"
                for m in item["metrics"]
            )
        cap = (
            f"<p class='cap'>{html.escape(item['caption'])}</p>"
            if item.get("caption")
            else ""
        )
        cards.append(
            f"<section class='card'><h3>{html.escape(item['title'])}</h3>{body}{cap}</section>"
        )
    return (
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font-family:Inter,Segoe UI,sans-serif;color:#1F2A37;margin:24px}"
        "h1{margin:0}.sub{color:#52606D;margin:4px 0 20px}"
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}"
        ".card{border:1px solid #E8ECF0;border-radius:8px;padding:12px;break-inside:avoid}"
        ".card h3{margin:0 0 8px;font-size:1rem}.cap{color:#52606D;font-size:.8rem}"
        ".kpi{display:inline-block;margin-right:24px}.kpi b{display:block;font-size:1.6rem}"
        "table{border-collapse:collapse;width:100%;font-size:.8rem}"
        "td,th{border-bottom:1px solid #E8ECF0;padding:4px 6px;text-align:left}"
        "@media print{body{margin:0}}</style></head><body>"
        f"<h1>{html.escape(title)}</h1><div class='sub'>{html.escape(subtitle)}</div>"
        f"<div class='grid'>{''.join(cards)}</div></body></html>"
    )


# ==============================================================================
# SCRAPING : requêtes longues exécutées en arrière-plan
# ==============================================================================
def _post_scrape_request(path: str, payload: dict, token: str) -> dict:
    response = requests.post(
        f"{FASTAPI_URL}{path}",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=(10, 3600),
    )
    try:
        data = response.json()
    except ValueError:
        data = {}
    return {
        "status_code": response.status_code,
        "data": data,
        "text": response.text,
    }


def _scrape_operation_body() -> None:
    future: Future | None = st.session_state.get("scrape_future")
    if future is None:
        return

    operation = st.session_state.get("scrape_operation", "scraping")
    session_id = st.session_state.get("session_id")

    if not future.done():
        st.info("Traitement en cours. Vous pouvez l'arrêter sans fermer la page.")
        if st.button(
            "Arrêter le traitement",
            key="cancel_scrape_operation",
            disabled=st.session_state.get("scrape_cancel_requested", False),
        ):
            if session_id:
                try:
                    cancel_response = requests.post(
                        f"{FASTAPI_URL}/ingest/discover-cancel/{session_id}",
                        headers=_auth_headers(),
                        timeout=5,
                    )
                    if cancel_response.ok:
                        st.session_state.scrape_cancel_requested = True
                        st.info(
                            "Arrêt demandé. Le serveur termine l'opération en cours."
                        )
                    else:
                        st.error(
                            f"Impossible d'arrêter le traitement "
                            f"({cancel_response.status_code}) : {cancel_response.text}"
                        )
                except requests.RequestException as exc:
                    st.error(
                        f"Impossible de joindre l'API pour arrêter le traitement : {exc}"
                    )
            return

        if st.session_state.get("scrape_cancel_requested"):
            st.caption("Annulation en cours…")
        return

    try:
        result = future.result()
        status_code = result["status_code"]
        data = result["data"]
        if status_code != 200:
            message = data.get("message") or data.get("detail") or result["text"]
            st.session_state.scrape_notice = (
                "error",
                f"Erreur API ({status_code}) : {message}",
            )
        elif data.get("status") == "cancelled":
            if operation == "start":
                st.session_state.session_id = None
                st.session_state.discovered_links = []
            st.session_state.scrape_notice = (
                "warning",
                data.get("message", "Traitement annulé."),
            )
        elif operation == "start":
            st.session_state.session_id = data.get("session_id")
            st.session_state.discovered_links = data.get("discovered_links", [])
            st.session_state.scrape_notice = (
                "success",
                f"Session initialisée : {len(st.session_state.discovered_links)} liens découverts.",
            )
        elif operation == "continue":
            st.session_state.discovered_links = data.get("discovered_links", [])
            st.session_state.scrape_notice = (
                "success",
                data.get("message", "Nouveau niveau exploré."),
            )
        else:
            st.session_state.scrape_last_result = data
            st.session_state.session_id = None
            st.session_state.discovered_links = []
            st.session_state.scrape_notice = (
                "success",
                "Exploration terminée et traitement finalisé.",
            )
    except Exception as exc:
        st.session_state.scrape_notice = (
            "error",
            f"Erreur pendant le traitement : {exc}",
        )
    finally:
        st.session_state.scrape_future = None
        st.session_state.scrape_operation = None
        st.session_state.scrape_cancel_requested = False

    st.rerun()


def render_scrape_operation() -> None:
    """Le polling toutes les secondes n'est actif QUE pendant un traitement.
    Avant : la page se rafraîchissait en permanence, même au repos."""
    pending = st.session_state.get("scrape_future") is not None
    st.fragment(_scrape_operation_body, run_every=1 if pending else None)()


def reset_voice_state() -> None:
    st.session_state.voice_transcript = ""
    st.session_state.voice_audio_hash = None
    st.session_state.voice_audio_key += 1
    st.session_state.voice_text_key += 1
    st.session_state.voice_retry_count = 0
    st.session_state.voice_requires_text = False


# code pour la gestion de la connexion avec la base de données
# ==============================================================================
# BASE DE DONNÉES : helpers (appels API, modal de connexion)
# ==============================================================================
def _auth_headers() -> dict:
    token = st.session_state.get("auth_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _db_api(method: str, path: str, **kwargs) -> tuple[bool, object]:
    """Appelle l'API d'ingestion BD. Retourne (succès, données ou message d'erreur)."""
    try:
        res = requests.request(
            method,
            f"{FASTAPI_URL}{path}",
            headers=_auth_headers(),
            timeout=(10, 3600),
            **kwargs,
        )
    except requests.RequestException as e:
        return False, f"Erreur de communication avec l'API : {e}"
    try:
        data = res.json()
    except ValueError:
        data = {}
    if res.status_code >= 400:
        detail = data.get("detail") if isinstance(data, dict) else None
        message = (
            detail
            if isinstance(detail, str)
            else (res.text or f"Erreur {res.status_code}")
        )
        return False, message
    return True, data


@st.cache_data(ttl=30, show_spinner=False)
def _cached_get(path: str, tenant_id: str, schema: str = "", table: str = ""):
    params = {"tenant_id": tenant_id}
    if schema:
        params["schema"] = schema
    if table:
        params["table"] = table
    return _db_api("GET", path, params=params)


def _db_get(path: str, tenant_id: str, schema: str = "", table: str = ""):
    ok, data = _cached_get(path, tenant_id, schema, table)
    if not ok:
        _cached_get.clear()  # on ne garde jamais une erreur en cache
    return ok, data


def _db_columns(conn_id: str, tenant_id: str, schema: str, table: str):
    ok, data = _db_get(
        f"/ingest/db/{conn_id}/columns", tenant_id, schema=schema, table=table
    )
    if not ok:
        st.error(data)
        return None
    return data


def _source_picker(
    label: str, name: str, refs: list, kp: str, default: str = "", placeholder: str = ""
) -> dict:
    """Valeur fixe OU colonne de la base (valeur propre à chaque ligne)."""
    mode = st.radio(
        label, ["Valeur fixe", "Colonne"], horizontal=True, key=f"db_{name}_mode_{kp}"
    )
    if mode == "Colonne":
        return {
            "column": st.selectbox(
                f"Colonne « {label} »",
                refs,
                key=f"db_{name}_col_{kp}",
                label_visibility="collapsed",
            )
        }
    return {
        "value": st.text_input(
            f"Valeur « {label} »",
            value=default,
            placeholder=placeholder,
            key=f"db_{name}_val_{kp}",
            label_visibility="collapsed",
        ).strip()
    }


@st.dialog("Connexion PostgreSQL")
def postgres_dialog(tenant_id: str) -> None:
    with st.form("pg_form"):
        name = st.text_input("Nom de la connexion", placeholder="Ex. : ERP production")
        c1, c2 = st.columns([3, 1])
        host = c1.text_input("Hôte", value="localhost")
        port = c2.number_input("Port", value=5432, min_value=1, max_value=65535, step=1)
        database = st.text_input("Base de données")
        username = st.text_input("Utilisateur")
        password = st.text_input("Mot de passe", type="password")
        show_all = st.checkbox(
            "Afficher toutes les tables",
            help="Cochée : tous les schémas. Sinon : uniquement le schéma « public ».",
        )
        b1, b2 = st.columns(2)
        test = b1.form_submit_button("Tester la connexion", use_container_width=True)
        save = b2.form_submit_button(
            "Enregistrer", type="primary", use_container_width=True
        )

    if not (test or save):
        return
    if not (host.strip() and database.strip() and username.strip() and password):
        st.error("Renseignez l'hôte, la base, l'utilisateur et le mot de passe.")
        return

    payload = {
        "host": host.strip(),
        "port": int(port),
        "database": database.strip(),
        "username": username.strip(),
        "password": password,
    }
    if test:
        with st.spinner("Test de la connexion…"):
            ok, res = _db_api("POST", "/ingest/db/test", data=payload)
        if ok:
            st.success("Connexion réussie.")
        else:
            st.error(res)
        return

    if not name.strip():
        st.error("Donnez un nom à la connexion pour l'enregistrer.")
        return

    with st.spinner("Test puis enregistrement…"):
        ok, res = _db_api(
            "POST",
            "/ingest/db/connections",
            data={
                **payload,
                "tenant_id": tenant_id,
                "name": name.strip(),
                "show_all_schemas": str(show_all).lower(),
            },
        )
        if not ok:
            st.error(res)
            return

    # Nettoyage du cache et mise à jour de l'état de session
    _cached_get.clear()
    st.session_state.db_last_saved = res["connection_id"]
    st.session_state.db_notice = f"Connexion « {name.strip()} » enregistrée."

    st.rerun()  # Ferme le modal et rafraîchit la page proprement


# ==============================================================================
# CHART CONFIG
# ==============================================================================

PALETTES = {
    "Par défaut": px.colors.qualitative.Plotly,
    "Pastel": px.colors.qualitative.Pastel,
    "Vif": px.colors.qualitative.Bold,
    "Varié": px.colors.qualitative.Dark24,
    "Bleus": px.colors.sequential.Blues[3:],
    "Verts": px.colors.sequential.Greens[3:],
    "Chaud": px.colors.sequential.Sunset,
}
CHART_TYPES = {
    "vertical_bar": "Barres verticales",
    "horizontal_bar": "Barres horizontales",
    "line": "Courbe",
    "area": "Aire",
    "donut": "Donut",
    "pie": "Camembert",
    "table": "Tableau",
}
DEFAULT_TOP_N = 10


def _short(col: str) -> str:
    return col.split(".")[-1]


def _prepare(data: list, cfg: dict):
    df = pd.DataFrame(data)
    measures = [m for m in cfg.get("measures", []) if m in df.columns]
    for m in measures:  # Cube renvoie souvent les nombres sous forme de chaînes
        df[m] = pd.to_numeric(df[m], errors="coerce")
    dims = [d for d in cfg.get("dimensions", []) if d in df.columns]

    time_col = None
    for td in cfg.get("timeDimensions", []):
        cands = [c for c in df.columns if c == td or c.startswith(td + ".")]
        if cands:
            time_col = cands[-1]
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col])
            break
    return df, measures, dims, time_col


def render_interactive_chart(cfg: dict, data: list, key: str) -> dict | None:
    print("data render interactive chart")
    if not data or cfg.get("type") == "empty":
        return
    df, measures, dims, time_col = _prepare(data, cfg)

    # --- Chiffre clé / pas de mesure exploitable ---
    if cfg.get("type") == "kpi_card" or not measures:
        if measures:
            for col, m in zip(st.columns(len(measures)), measures):
                col.metric(_short(m), f"{df[m].iloc[0]:,.2f}".replace(",", " "))
            return {
                "kind": "kpi",
                "metrics": [
                    {
                        "label": _short(m),
                        "value": f"{df[m].iloc[0]:,.2f}".replace(",", " "),
                    }
                    for m in measures
                ],
            }
        st.dataframe(df, use_container_width=True)
        return _table_snapshot(df)

    cat_col = dims[0] if dims else None
    x_col = time_col or cat_col
    color_col = cat_col if time_col else (dims[1] if len(dims) > 1 else None)
    if x_col is None:
        st.dataframe(df, use_container_width=True)
        return _table_snapshot(df)

    default_type = cfg.get("type") if cfg.get("type") in CHART_TYPES else "table"
    type_keys = list(CHART_TYPES)

    # --- Contrôles ---
    order_cat = None
    with st.expander("⚙️ Personnaliser le graphique"):
        c1, c2, c3 = st.columns(3)
        ctype = c1.selectbox(
            "Type de graphique",
            type_keys,
            index=type_keys.index(default_type),
            format_func=CHART_TYPES.get,
            key=f"{key}_type",
        )
        palette_name = c2.selectbox("Palette", list(PALETTES), key=f"{key}_pal")
        if len(measures) > 1:
            measure = c3.selectbox(
                "Mesure", measures, format_func=_short, key=f"{key}_m"
            )
        else:
            measure = measures[0]

        # Filtre date : bornes = min/max des données affichées
        if time_col and df[time_col].nunique() > 1:
            dmin = df[time_col].min().to_pydatetime()
            dmax = df[time_col].max().to_pydatetime()
            start, end = st.slider(
                "Période",
                min_value=dmin,
                max_value=dmax,
                value=(dmin, dmax),
                format="DD/MM/YYYY",
                key=f"{key}_dates",
            )
            df = df[(df[time_col] >= start) & (df[time_col] <= end)]

        # Filtre liste : top N / bottom N
        if cat_col and df[cat_col].nunique() > 1:
            n_unique = df[cat_col].nunique()
            f1, f2 = st.columns([3, 2])
            top_n = f1.slider(
                "Nombre d'éléments affichés",
                1,
                n_unique,
                min(n_unique, DEFAULT_TOP_N),
                key=f"{key}_topn",
            )
            mode = f2.radio(
                "Tri",
                ["Plus grands", "Plus petits"],
                horizontal=True,
                key=f"{key}_sort",
            )
            totals = (
                df.groupby(cat_col)[measure]
                .sum()
                .sort_values(ascending=(mode == "Plus petits"))
            )
            order_cat = list(totals.head(top_n).index)
            df = df[df[cat_col].isin(order_cat)]

    if df.empty:
        st.info("Aucune donnée sur cette sélection.")
        return

    group = [c for c in (x_col, color_col) if c]
    df = df.groupby(group, as_index=False)[measure].sum()
    if x_col == time_col:
        df = df.sort_values(x_col)

    # --- Construction de la figure ---
    palette = PALETTES[palette_name]
    labels = {c: _short(c) for c in df.columns}
    title = cfg.get("title")
    x_lab = cfg.get("xAxisLabel") or labels[x_col]
    y_lab = cfg.get("yAxisLabel") or labels[measure]
    horizontal = ctype == "horizontal_bar"
    cats = (order_cat[::-1] if horizontal else order_cat) if order_cat else None
    cat_order = {x_col: cats} if (cats and x_col == cat_col) else {}
    common = dict(color_discrete_sequence=palette, labels=labels, title=title)

    if ctype == "table":
        # 1. Copie du DataFrame pour formater l'affichage sans casser les données brutes
        display_df = df.copy()

        # 2. Formatage des colonnes numériques (séparateur de milliers, 2 décimales)
        for col in display_df.select_dtypes(include=["number"]).columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:,.2f}".replace(",", " ") if pd.notnull(x) else ""
            )

        # 3. Conversion propre en HTML avec un style CSS professionnel
        # En-tête aux couleurs de ta charte (ex: SLATE_900 ou ACCENT), alternance de lignes, texte centré
        html_table = display_df.to_html(
            classes="styled-table", index=False, escape=False
        )

        styled_html = f"""
        <style>
            .styled-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
                font-size: 0.9rem;
                font-family: 'Inter', sans-serif;
                background-color: #FFFFFF;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .styled-table th {{
                background-color: #1F2A37; /* SLATE_900 */
                color: #FFFFFF;
                text-align: center !important;
                padding: 12px 15px;
                font-weight: 600;
            }}
            .styled-table td {{
                padding: 10px 15px;
                text-align: center !important; /* Texte centré par défaut */
                border-bottom: 1px solid #E8ECF0;
                color: #334155;
            }}
            .styled-table tbody tr:hover {{
                background-color: #F8FAFC;
            }}
            /* Style pour les chiffres négatifs (si détectés dans le texte ou format) */
            .negative-val {{
                color: #DC2626 !important;
                font-weight: bold;
            }}
        </style>
        <div style="overflow-x: auto;">
            {html_table}
        </div>
        """

        # 4. Post-traitement HTML rapide pour colorer en rouge/gras les valeurs négatives
        # (Recherche les signes moins dans le rendu HTML pour leur appliquer la classe CSS)
        styled_html = styled_html.replace(">-", "><span class='negative-val'>-")
        styled_html = styled_html.replace(
            "</span></p>", "</span>"
        )  # Nettoyage optionnel selon pandas
        # Astuce propre pour fermer le span des nombres négatifs s'ils contiennent des chiffres après
        import re

        styled_html = re.sub(
            r">(-[\d\s.,]+)<", r'><span class="negative-val">\1</span><', styled_html
        )

        st.markdown(styled_html, unsafe_allow_html=True)
        return _table_snapshot(df)
    if ctype in ("donut", "pie"):
        pie_df = df.groupby(x_col, as_index=False)[measure].sum()
        fig = px.pie(
            pie_df,
            names=x_col,
            values=measure,
            hole=0.45 if ctype == "donut" else 0,
            **common,
        )
    elif ctype in ("line", "area"):
        fn = px.line if ctype == "line" else px.area
        extra = {"markers": True} if ctype == "line" else {}
        fig = fn(
            df,
            x=x_col,
            y=measure,
            color=color_col,
            category_orders=cat_order,
            **common,
            **extra,
        )
    else:
        color = color_col or (x_col if x_col != time_col else None)
        fig = px.bar(
            df,
            x=measure if horizontal else x_col,
            y=x_col if horizontal else measure,
            color=color,
            orientation="h" if horizontal else "v",
            barmode="group",
            category_orders=cat_order,
            **common,
        )
        if color == x_col:
            fig.update_layout(showlegend=False)

    if ctype not in ("donut", "pie"):
        fig.update_layout(
            xaxis_title=y_lab if horizontal else x_lab,
            yaxis_title=x_lab if horizontal else y_lab,
        )
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_fig")
    if cfg.get("reason"):
        st.caption(f"💡 Graphique suggéré : {cfg['reason']}")
    return {
        "kind": "figure",
        "fig_json": pio.to_json(fig),
        "caption": cfg.get("reason") or "",
    }


# ==============================================================================
# AUTHENTIFICATION
# ==============================================================================
if not st.session_state.get("auth_user"):
    st.title("Connexion à l’assistant ERP")
    st.caption("Connectez-vous avec les informations fournies par l’administrateur.")
    with st.container(border=True):
        with st.form("login_form"):
            login_tenant = st.text_input("Entreprise (identifiant du tenant)")
            login_name = st.text_input("Nom")
            login_surname = st.text_input("Prénom")
            login_password = st.text_input("Mot de passe", type="password")
            login_submitted = st.form_submit_button(
                "Se connecter", type="primary", width="stretch"
            )

    if login_submitted:
        if not all(
            value.strip()
            for value in (login_tenant, login_name, login_surname, login_password)
        ):
            st.error("Renseignez l’entreprise, le prénom, le nom et le mot de passe.")
        else:
            try:
                print(
                    "user informations",
                    login_name,
                    login_tenant,
                    login_surname,
                    login_password,
                )
                user = authenticate_user(
                    login_tenant, login_name, login_surname, login_password
                )
                if not user:
                    st.error("Informations de connexion invalides.")
                else:
                    token = create_access_token(user)
                    st.session_state.auth_user = user
                    st.session_state.auth_token = token
                    st.session_state.messages = []
                    st.rerun()
            except SQLAlchemyError:
                logger.exception("Échec de l’authentification PostgreSQL")
                st.error(
                    "Impossible de vérifier le compte. Vérifiez la connexion PostgreSQL "
                    "et l’existence de la table users.users_tests."
                )
            except RuntimeError as exc:
                st.error(str(exc))
    st.stop()

auth_user = st.session_state.auth_user
is_admin = auth_user["role"] == "admin"
tenant_id = auth_user["tenant_id"]
menu_options = ["Assistant", "Mon dashboard"]
if is_admin:
    menu_options.extend(
        [
            "Gestion des utilisateurs",
            "Ingestion de documents",
            "Exploration web",
            "Base de données",
            "Couche sémantique LanceDB",
        ]
    )
if st.session_state.get("module_navigation") not in menu_options:
    st.session_state.module_navigation = "Assistant"

# ---------------------------------
# ==============================================================================
# BARRE LATÉRALE
# ==============================================================================
with st.sidebar:
    if LOGO_WHITE.exists():
        st.image(str(LOGO_WHITE), width=170)
    else:
        st.markdown("###  ISNOV")
    st.markdown(
        "<div class='sidebar-caption' style='margin-top:.6rem'>Assistant ERP<br>"
        "Environnement de test</div>",
        unsafe_allow_html=True,
    )
    with st.popover(
        f":material/account_circle: {auth_user['name'][:1].upper()}",
        help="Profil et configuration",
        width="content",
    ):
        st.markdown(f"**{auth_user['name']} {auth_user['surname']}**")
        st.caption(f"{tenant_id} · {auth_user['role']}")
        if is_admin and st.button(
            "Configurer la base analytique",
            key="open_analytics_db_settings",
            width="stretch",
        ):
            postgres_dialog(tenant_id)
        if st.button("Se déconnecter", key="logout_button", width="stretch"):
            user_state_keys = {
                "auth_user",
                "auth_token",
                "messages",
                "module_navigation",
                "assistant_input_type",
                "session_id",
                "discovered_links",
            }
            user_state_prefixes = ("voice_", "scrape_", "db_", "upload_", "bi_", "dash")
            for key in list(st.session_state):
                if key in user_state_keys or key.startswith(user_state_prefixes):
                    st.session_state.pop(key, None)
            st.rerun()
    st.markdown("<div class='sidebar-section'>Navigation</div>", unsafe_allow_html=True)
    menu = st.radio(
        "Module",
        menu_options,
        key="module_navigation",
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        f"<div class='sidebar-caption'>API : {FASTAPI_URL}</div>",
        unsafe_allow_html=True,
    )

# ==============================================================================
# 0. GESTION DES UTILISATEURS (administrateur)
# ==============================================================================
if menu == "Gestion des utilisateurs" and is_admin:
    page_header(
        "Gestion des utilisateurs",
        "Créez les comptes de votre entreprise et attribuez leur rôle.",
    )
    st.caption(f"Entreprise / tenant : **{tenant_id}**")
    with st.container(border=True):
        st.subheader("Créer un utilisateur")
        with st.form("create_user_form", clear_on_submit=True):
            first_name = st.text_input("Prénom", max_chars=100)
            surname = st.text_input("Nom", max_chars=100)
            role = st.selectbox(
                "Rôle",
                ["user", "bi_analyst", "admin"],
                format_func=lambda value: {
                    "user": "Client",
                    "bi_analyst": "Analyste BI",
                    "admin": "Administrateur",
                }[value],
            )
            new_password = st.text_input("Mot de passe temporaire", type="password")
            confirm_password = st.text_input(
                "Confirmer le mot de passe", type="password"
            )
            create_submitted = st.form_submit_button(
                "Créer le compte", type="primary", width="stretch"
            )

    if create_submitted:
        if new_password != confirm_password:
            st.error("Les mots de passe ne correspondent pas.")
        else:
            try:
                created_user = create_tenant_user(
                    tenant_id, first_name, surname, new_password, role
                )
                st.success(
                    f"Compte créé pour {created_user['name']} "
                    f"{created_user['surname']} ({created_user['role']})."
                )
            except ValueError as exc:
                st.error(str(exc))
            except SQLAlchemyError:
                logger.exception("Échec de création d’un utilisateur")
                st.error(
                    "Impossible de créer le compte. Vérifiez la table users.users_tests "
                    "et les droits PostgreSQL."
                )

    try:
        users = list_tenant_users(tenant_id)
        st.subheader("Utilisateurs de l’entreprise")
        st.dataframe(
            [
                {
                    "Prénom": user["name"],
                    "Nom": user["surname"],
                    "Rôle": user["role"],
                }
                for user in users
            ],
            width="stretch",
            hide_index=True,
        )
    except SQLAlchemyError:
        logger.exception("Échec de lecture des utilisateurs")
        st.error("Impossible de charger les utilisateurs de cette entreprise.")

# ==============================================================================
# DASHBOARD PERSONNEL
# ==============================================================================
elif menu == "Mon dashboard":
    items = dashboard_items()
    st.session_state.setdefault("dashboard_title", "Mon dashboard")
    dash_title = st.session_state.dashboard_title.strip() or "Mon dashboard"
    dash_sub = f"{tenant_id} · {datetime.now():%d/%m/%Y}"

    st.markdown(
        f"<div class='page-title'>{html.escape(dash_title)}</div>"
        f"<div class='page-sub'>{html.escape(dash_sub)}</div>"
        f"<div class='page-rule'></div>",
        unsafe_allow_html=True,
    )

    with st.container(key="dash_toolbar"):  # masqué à l'impression
        t1, t2 = st.columns([3, 1])
        t1.text_input("Titre du dashboard", key="dashboard_title")
        t2.selectbox("Colonnes", [1, 2], index=1, key="dashboard_cols")
        b1, b2, b3 = st.columns(3)
        if b1.button(
            "🖨️ Imprimer", key="dashboard_print", type="primary", disabled=not items
        ):
            components.html("<script>window.parent.print();</script>", height=0)
        b2.download_button(
            "⬇️ Export HTML imprimable",
            data=build_dashboard_html(items, dash_title, dash_sub) if items else "",
            file_name="dashboard.html",
            mime="text/html",
            key="dashboard_export",
            disabled=not items,
        )
        b3.button(
            "Vider le dashboard",
            key="dashboard_clear",
            on_click=dashboard_clear,
            disabled=not items,
        )

    if not items:
        st.info(
            "Votre dashboard est vide. Posez une question à l'assistant, puis cliquez sur "
            "« Ajouter au dashboard » sous le graphique obtenu."
        )
    else:
        n_cols = st.session_state.dashboard_cols
        grid = st.columns(n_cols)
        for idx, item in enumerate(items):
            with grid[idx % n_cols]:
                with st.container(border=True):
                    st.markdown(f"**{item['title']}**")
                    render_snapshot(item, key=f"dash_{item['id']}")
                    with st.container(key=f"dash_actions_{item['id']}"):
                        a1, a2, a3, a4 = st.columns(4)
                        a1.button(
                            "↑",
                            key=f"dash_up_{item['id']}",
                            on_click=dashboard_move,
                            args=(item["id"], -1),
                            disabled=idx == 0,
                        )
                        a2.button(
                            "↓",
                            key=f"dash_down_{item['id']}",
                            on_click=dashboard_move,
                            args=(item["id"], 1),
                            disabled=idx == len(items) - 1,
                        )
                        with a3.popover("✏️"):
                            st.text_input(
                                "Titre",
                                value=item["title"],
                                key=f"dash_title_{item['id']}",
                            )
                            st.button(
                                "Enregistrer",
                                key=f"dash_ren_{item['id']}",
                                on_click=dashboard_rename,
                                args=(item["id"],),
                            )
                        a4.button(
                            "🗑️",
                            key=f"dash_del_{item['id']}",
                            on_click=dashboard_remove,
                            args=(item["id"],),
                        )

# ==============================================================================
# 1. ASSISTANT (texte + transcription vocale validée en 2 essais)
# ==============================================================================
elif menu == "Assistant":
    page_header(
        "Assistant ERP",
        "Posez votre question par écrit ou envoyez un message vocal. "
        "Pour le vocal, validez la transcription (2 essais maximum), puis saisissez "
        "votre demande par écrit si elle ne convient toujours pas.",
    )

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("voice_retry_count", 0)
    st.session_state.setdefault("voice_transcript", "")
    st.session_state.setdefault("voice_audio_hash", None)
    st.session_state.setdefault("voice_audio_key", 0)  # change -> vide le widget micro
    st.session_state.setdefault("voice_text_key", 0)  # change -> vide la zone de texte
    st.session_state.setdefault("voice_requires_text", False)
    st.session_state.setdefault("voice_notice", None)
    st.session_state.setdefault("assistant_input_type", "Saisie texte")

    for message_index, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            content = msg["content"]

            # Détection basique ou par blocs de code de la syntaxe Mermaid
            is_mermaid = "```mermaid" in content or content.strip().startswith(
                ("graph ", "flowchart ")
            )
            if is_mermaid:
                # Extraction du code Mermaid si encapsulé dans des balises markdown
                if "```mermaid" in content:
                    parts = content.split("```mermaid")
                    text_before = parts[0]
                    mermaid_code = parts[1].split("```")[0].strip()
                    text_after = (
                        parts[1].split("```")[1]
                        if len(parts[1].split("```")) > 1
                        else ""
                    )

                    if text_before.strip():
                        st.markdown(text_before)

                    # Rendu du diagramme Mermaid interactif
                    st_mermaid(mermaid_code)

                    if text_after.strip():
                        st.markdown(text_after)
                else:
                    # Si le message est directement du mermaid brut
                    mermaid_code = content.strip()
                    st_mermaid(mermaid_code)

            else:
                st.markdown(content)

            if msg.get("chart_config") and msg.get("data"):
                snapshot = render_interactive_chart(
                    msg["chart_config"], msg["data"], key=f"chart_{message_index}"
                )
                # Le bouton est évalué APRÈS le graphique : le snapshot reflète
                # donc les réglages courants (type, palette, filtres).
                if snapshot and st.button(
                    "📌 Ajouter au dashboard", key=f"dash_add_{message_index}"
                ):
                    question = (
                        st.session_state.messages[message_index - 1]["content"]
                        if message_index > 0
                        else ""
                    )
                    title = (
                        msg["chart_config"].get("title") or question[:80] or "Graphique"
                    )
                    if dashboard_add(snapshot, title):
                        st.toast(
                            f"Ajouté au dashboard ({len(dashboard_items())} élément(s)).",
                            icon="📌",
                        )
                    else:
                        st.toast("Ce graphique est déjà dans le dashboard.", icon="ℹ️")
            if msg["role"] == "assistant":
                msg_id = hashlib.md5(content.encode()).hexdigest()
                # On récupère la langue associée au message (par défaut 'fr' si saisie texte directe)
                msg_lang = msg.get("language", "fr")

                if st.button(
                    f"🔊 Écouter ({msg_lang.upper()})",
                    key=f"play_{message_index}_{msg_id}",
                ):
                    with st.spinner("Génération de la voix..."):
                        try:
                            res = requests.post(
                                f"{FASTAPI_URL}/text-to-speech",
                                data={
                                    "text": content,
                                    "language": msg_lang,  # <-- Envoi de la bonne langue à l'API
                                },
                                timeout=3000,
                            )
                            if res.status_code == 200:
                                st.audio(res.content, format="audio/wav", autoplay=True)
                            else:
                                st.error("Impossible de lire ce texte.")
                        except Exception as e:
                            st.sidebar.error(f"Erreur TTS : {e}")

    if st.session_state.voice_requires_text:
        st.session_state.assistant_input_type = "Saisie texte"

    input_type = st.radio(
        "Mode d'entrée",
        ["Saisie texte", "Message vocal"],
        horizontal=True,
        key="assistant_input_type",
        disabled=st.session_state.voice_requires_text,
    )

    if st.session_state.voice_notice:
        level, message = st.session_state.voice_notice
        if level == "warning":
            st.warning(message)
        else:
            st.info(message)
        st.session_state.voice_notice = None

    if input_type == "Saisie texte":
        if prompt := st.chat_input("Écrivez votre question ici…"):
            with st.spinner("Analyse en cours…"):
                try:
                    res = requests.post(
                        f"{FASTAPI_URL}/chat",
                        data={"raw_query": prompt, "input_mode": "text"},
                        headers=_auth_headers(),
                        timeout=(10, 3600),
                    )
                    if res.status_code == 200:
                        payload = res.json()
                        answer = payload.get("answer") or "Réponse vide."
                        st.session_state.messages.append(
                            {"role": "user", "content": prompt}
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "chart_config": payload.get("chart_config"),
                                "data": payload.get("data"),
                            }
                        )
                        reset_voice_state()
                        st.rerun()
                    else:
                        st.error(f"Erreur API ({res.status_code}) : {res.text}")
                except requests.RequestException as e:
                    st.error(f"Impossible de joindre l'API : {e}")

    else:
        with st.container(border=True):
            st.subheader("Message vocal")
            audio = st.audio_input(
                "Appuyez pour enregistrer, puis arrêtez",
                key=f"voice_audio_{st.session_state.voice_audio_key}",
            )

            if audio is None:
                st.session_state.voice_audio_hash = None
            else:
                audio_bytes = audio.getvalue()
                digest = hashlib.md5(audio_bytes).hexdigest()

                # Nouvel enregistrement : on repart d'une transcription vide
                if digest != st.session_state.voice_audio_hash:
                    st.session_state.voice_audio_hash = digest
                    st.session_state.voice_transcript = ""
                    st.session_state.voice_text_key += 1

                # Étape 1 : transcription
                if not st.session_state.voice_transcript and st.button(
                    "Transcrire", type="primary"
                ):
                    with st.spinner("Transcription en cours…"):
                        clean_file_path = None
                        try:
                            # Étape A : préparer un fichier audio propre pour la transcription.
                            clean_file_path = clean_and_save_audio(audio_bytes)

                            # Étape B : transmettre le contenu du fichier à l'API.
                            with open(clean_file_path, "rb") as clean_audio_file:
                                res = requests.post(
                                    f"{FASTAPI_URL}/transcribe",
                                    files={
                                        "file": (
                                            "voice.wav",
                                            clean_audio_file,
                                            "audio/wav",
                                        )
                                    },
                                    timeout=(10, 600),
                                )
                            if res.status_code == 200:
                                text = (res.json().get("text") or "").strip()
                                if text:
                                    st.session_state.voice_transcript = text
                                    st.rerun()
                                else:
                                    st.warning("Aucune parole détectée, réessayez.")
                            else:
                                st.error(f"Erreur API ({res.status_code}) : {res.text}")
                        except requests.RequestException as e:
                            st.error(f"Erreur de connexion : {e}")
                        finally:
                            if clean_file_path and os.path.exists(clean_file_path):
                                os.remove(clean_file_path)

            # Étape 2 : validation de la transcription avant envoi au chat normal
            if st.session_state.voice_transcript:
                edited = st.text_area(
                    "Transcription (modifiable)",
                    value=st.session_state.voice_transcript,
                    key=f"voice_text_{st.session_state.voice_text_key}",
                )
                col1, col2 = st.columns(2)

                with col1:
                    if st.button(
                        "Valider la transcription et exécuter",
                        type="primary",
                        use_container_width=True,
                        disabled=not edited.strip(),
                    ):
                        with st.spinner("Analyse de la transcription validée…"):
                            try:
                                res = requests.post(
                                    f"{FASTAPI_URL}/chat",
                                    data={
                                        "raw_query": edited.strip(),
                                        "input_mode": "text",
                                    },
                                    headers=_auth_headers(),
                                    timeout=(10, 3600),
                                )
                                if res.status_code == 200:
                                    payload = res.json()
                                    final_answer = (
                                        payload.get("answer") or "Réponse vide."
                                    )

                                    final_answer = (
                                        res.json().get("answer") or "Réponse vide."
                                    )
                                    detected_lang = st.session_state.get(
                                        "last_detected_lang", "fr"
                                    )
                                    st.session_state.messages.append(
                                        {"role": "user", "content": edited.strip()}
                                    )
                                    st.session_state.messages.append(
                                        {
                                            "role": "assistant",
                                            "content": final_answer,
                                            "language": detected_lang,
                                            "chart_config": payload.get("chart_config"),
                                            "data": payload.get("data"),
                                        }
                                    )
                                    reset_voice_state()
                                    st.rerun()
                                else:
                                    st.error(
                                        f"Erreur API ({res.status_code}) : {res.text}"
                                    )
                            except requests.RequestException as e:
                                st.error(f"Erreur de connexion : {e}")

                with col2:
                    if st.button("Refuser et réessayer", use_container_width=True):
                        st.session_state.voice_retry_count += 1

                        if st.session_state.voice_retry_count >= 2:
                            st.session_state.voice_requires_text = True
                            st.session_state.voice_notice = (
                                "warning",
                                (
                                    "Deux transcriptions ont été refusées. "
                                    "Veuillez saisir votre demande par écrit."
                                ),
                            )
                        else:
                            st.session_state.voice_notice = (
                                "info",
                                (
                                    "Transcription refusée. Enregistrez un nouvel audio "
                                    "pour votre deuxième et dernier essai vocal."
                                ),
                            )
                        st.session_state.voice_transcript = ""
                        st.session_state.voice_audio_hash = None
                        st.session_state.voice_audio_key += 1
                        st.session_state.voice_text_key += 1
                        st.rerun()


# ==============================================================================
# 2. INGESTION DE DOCUMENTS (PDF, CSV, TXT, MD, PPTX)
# ==============================================================================
elif menu == "Ingestion de documents":
    page_header(
        "Ingestion de documents",
        "Ajoutez vos documents (PDF, CSV, TXT, MD, PPTX) pour les indexer dans la base vectorielle.",
    )

    with st.container(border=True):
        st.caption(f"Entreprise / tenant : {tenant_id}")
        uploaded_files = st.file_uploader(
            "Choisir un ou plusieurs fichiers",
            type=["pdf", "csv", "txt", "md", "pptx"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("Lancer l'ingestion", type="primary"):
            for uploaded_file in uploaded_files:
                with st.spinner(
                    f"Conversion et indexation de {uploaded_file.name} en cours…"
                ):
                    try:
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type,
                            )
                        }
                        res = requests.post(
                            f"{FASTAPI_URL}/ingest/upload",
                            files=files,
                            data={"tenant_id": tenant_id},
                            headers=_auth_headers(),
                            timeout=(10, 3600),
                        )

                        if res.status_code == 202:
                            try:
                                result_data = res.json()
                            except ValueError:
                                st.error(
                                    f"Réponse invalide de l'API pour "
                                    f"**{uploaded_file.name}** : {res.text}"
                                )
                            else:
                                st.success(
                                    f"Fichier **{result_data.get('filename')}** "
                                    "indexé avec succès."
                                )
                                st.json(result_data, expanded=False)
                        else:
                            st.error(
                                f"Erreur d'ingestion de **{uploaded_file.name}** "
                                f"({res.status_code}) : {res.text}"
                            )
                    except requests.RequestException as e:
                        st.error(
                            f"Erreur de communication avec l'API pour "
                            f"**{uploaded_file.name}** : {e}"
                        )


# ==============================================================================
# 3. EXPLORATION WEB (découverte interactive en 2 étapes)
# ==============================================================================
elif menu == "Exploration web":
    page_header(
        "Exploration web",
        "Explorez une documentation en ligne niveau par niveau, "
        "puis finalisez l'ingestion dans LanceDB.",
    )

    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("discovered_links", [])
    st.session_state.setdefault("scrape_future", None)
    st.session_state.setdefault("scrape_operation", None)
    st.session_state.setdefault("scrape_cancel_requested", False)
    st.session_state.setdefault("scrape_notice", None)
    st.session_state.setdefault("scrape_last_result", None)

    scrape_pending = st.session_state.scrape_future is not None
    render_scrape_operation()

    notice = st.session_state.scrape_notice
    if notice:
        level, message = notice
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.error(message)
        st.session_state.scrape_notice = None

    if st.session_state.scrape_last_result is not None:
        st.json(st.session_state.scrape_last_result, expanded=False)
        st.session_state.scrape_last_result = None

    with st.container(border=True):
        start_url = st.text_input(
            "URL de départ", placeholder="https://wiki.dolibarr.org"
        )

        if (
            st.button(
                "Démarrer l'exploration",
                type="primary",
                disabled=scrape_pending,
            )
            and start_url
        ):
            session_id = uuid.uuid4().hex
            st.session_state.session_id = session_id
            st.session_state.discovered_links = []
            st.session_state.scrape_operation = "start"
            st.session_state.scrape_cancel_requested = False
            st.session_state.scrape_future = SCRAPE_EXECUTOR.submit(
                _post_scrape_request,
                "/ingest/discover-start",
                {
                    "start_url": start_url,
                    "tenant_id": tenant_id,
                    "session_id": session_id,
                },
                st.session_state.auth_token,
            )
            st.rerun()

    # Commandes de session disponibles même après une annulation.
    if st.session_state.session_id:
        selected_urls = []
        with st.container(border=True):
            if st.session_state.discovered_links:
                st.subheader("Liens découverts")
                st.caption("Cochez les liens à approfondir.")
                for link in st.session_state.discovered_links:
                    if st.checkbox(
                        f"{link['label']} ({link['url']})",
                        key=link["url"],
                        disabled=scrape_pending,
                    ):
                        selected_urls.append(link["url"])
            else:
                st.info(
                    "Aucun nouveau lien à explorer. Vous pouvez finaliser la session."
                )

            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Continuer avec la sélection",
                    disabled=not selected_urls or scrape_pending,
                    use_container_width=True,
                ):
                    st.session_state.scrape_operation = "continue"
                    st.session_state.scrape_cancel_requested = False
                    st.session_state.scrape_future = SCRAPE_EXECUTOR.submit(
                        _post_scrape_request,
                        "/ingest/discover-continue",
                        {
                            "session_id": st.session_state.session_id,
                            "tenant_id": tenant_id,
                            "continue_exploration": True,
                            "selected_urls": selected_urls,
                        },
                        st.session_state.auth_token,
                    )
                    st.rerun()

            with col2:
                if st.button(
                    "Arrêter et finaliser l'ingestion",
                    type="primary",
                    disabled=scrape_pending,
                    use_container_width=True,
                ):
                    st.session_state.scrape_operation = "finalize"
                    st.session_state.scrape_cancel_requested = False
                    st.session_state.scrape_future = SCRAPE_EXECUTOR.submit(
                        _post_scrape_request,
                        "/ingest/discover-continue",
                        {
                            "session_id": st.session_state.session_id,
                            "tenant_id": tenant_id,
                            "continue_exploration": False,
                            "selected_urls": [],
                        },
                        st.session_state.auth_token,
                    )
                    st.rerun()

# ==============================================================================
# 4. BASE DE DONNÉES (connexion PostgreSQL → colonnes → ingestion LanceDB)
# ==============================================================================
elif menu == "Base de données":
    page_header(
        "Base de données",
        "Connectez une base PostgreSQL, choisissez les colonnes à indexer, "
        "puis ingérez-les dans la base vectorielle.",
    )
    st.session_state.setdefault("db_joins", [])
    st.session_state.setdefault("db_notice", None)
    st.session_state.setdefault("db_last_saved", None)

    if st.session_state.db_notice:
        st.success(st.session_state.db_notice)
        st.session_state.db_notice = None

    st.caption(f"Entreprise / tenant : **{tenant_id}**")

    # ---- 1. Connexion ---------------------------------------------------------
    with st.container(border=True):
        st.subheader("1. Connexion")
        b1, b2, b3 = st.columns(3)
        if b1.button("PostgreSQL", type="primary", use_container_width=True):
            postgres_dialog(tenant_id)
        b2.button("MySQL (bientôt)", disabled=True, use_container_width=True)
        b3.button("SQL Server (bientôt)", disabled=True, use_container_width=True)

        ok, res = _db_get("/ingest/db/connections", tenant_id)
        if not ok:
            st.error(res)
            st.stop()
        connections = res.get("connections", [])
        if not connections:
            st.info(
                "Aucune connexion enregistrée pour ce tenant. "
                "Cliquez sur PostgreSQL pour en créer une."
            )
            st.stop()
        ids = [c["connection_id"] for c in connections]
        names = {
            c["connection_id"]: f"{c['name']} — {c['database']}@{c['host']}"
            for c in connections
        }
        last = st.session_state.db_last_saved
        conn_id = st.selectbox(
            "Connexion enregistrée",
            ids,
            index=ids.index(last) if last in ids else 0,
            format_func=names.get,
        )

    # ---- 2. Table principale --------------------------------------------------
    with st.container(border=True):
        st.subheader("2. Table principale")
        ok, res = _db_get(f"/ingest/db/{conn_id}/schemas", tenant_id)
        if not ok:
            st.error(res)
            st.stop()
        schemas = res.get("schemas", [])
        if not schemas:
            st.warning("Aucun schéma accessible avec cette connexion.")
            st.stop()
        s1, s2 = st.columns(2)
        schema = s1.selectbox("Schéma", schemas)
        ok, res = _db_get(f"/ingest/db/{conn_id}/tables", tenant_id, schema=schema)
        if not ok:
            st.error(res)
            st.stop()
        table_names = [t["name"] for t in res.get("tables", [])]
        if not table_names:
            st.warning("Aucune table dans ce schéma.")
            st.stop()
        table = s2.selectbox("Table", table_names)

    # Changer de table principale réinitialise les jointures et les sélections.
    base_key = (conn_id, schema, table)
    if st.session_state.get("db_base_key") != base_key:
        st.session_state.db_base_key = base_key
        st.session_state.db_joins = []
    joins = st.session_state.db_joins
    kp = hashlib.md5(
        repr((base_key, [j["table"] for j in joins])).encode()
    ).hexdigest()[:8]

    base_cols = _db_columns(conn_id, tenant_id, schema, table)
    if base_cols is None:
        st.stop()
    tables_in_play = [{"table": table, "data": base_cols}]
    for j in joins:
        data = _db_columns(conn_id, tenant_id, j["schema"], j["table"])
        if data is None:
            st.stop()
        tables_in_play.append({"table": j["table"], "data": data})
    refs = [
        f"{t['table']}.{c['name']}"
        for t in tables_in_play
        for c in t["data"]["columns"]
    ]

    # ---- Jointures (optionnel) ------------------------------------------------
    with st.expander("Relier d'autres tables (optionnel)", expanded=bool(joins)):
        st.caption(
            "Reliez d'abord les tables, puis choisissez les colonnes plus bas "
            "(la sélection est réinitialisée à chaque changement de jointure)."
        )
        for j in joins:
            st.markdown(
                f"- `{j['left_table']}.{j['left_column']}` → `{j['table']}.{j['right_column']}`"
            )
        if joins and st.button("Retirer la dernière jointure"):
            joins.pop()
            st.rerun()

        used = {t["table"] for t in tables_in_play}
        j1, j2 = st.columns(2)
        j_schema = j1.selectbox(
            "Schéma de la table à relier", schemas, key=f"db_js_{kp}"
        )
        ok, res = _db_get(f"/ingest/db/{conn_id}/tables", tenant_id, schema=j_schema)
        candidates = (
            [t["name"] for t in res.get("tables", []) if t["name"] not in used]
            if ok
            else []
        )
        if not candidates:
            st.info("Aucune autre table disponible dans ce schéma.")
        else:
            j_table = j2.selectbox("Table à relier", candidates, key=f"db_jt_{kp}")
            j_data = _db_columns(conn_id, tenant_id, j_schema, j_table)
            if j_data:
                k1, k2 = st.columns(2)
                left_ref = k1.selectbox(
                    "Colonne déjà sélectionnée", refs, key=f"db_jl_{kp}"
                )
                right_col = k2.selectbox(
                    f"Colonne de « {j_table} »",
                    [c["name"] for c in j_data["columns"]],
                    key=f"db_jr_{kp}",
                )
                if st.button("Ajouter la jointure"):
                    left_table, _, left_col = left_ref.partition(".")
                    joins.append(
                        {
                            "schema": j_schema,
                            "table": j_table,
                            "left_table": left_table,
                            "left_column": left_col,
                            "right_column": right_col,
                        }
                    )
                    st.rerun()

    # ---- 3. Colonnes à indexer ------------------------------------------------
    with st.container(border=True):
        st.subheader("3. Colonnes à indexer")
        st.caption(f"Aperçu de la table principale « {table} » (3 lignes).")
        st.dataframe(base_cols["sample"], use_container_width=True)
        body_refs = st.multiselect(
            "Colonnes qui composent le texte indexé", refs, key=f"db_body_{kp}"
        )
        labels = {}
        for ref in body_refs:
            default_label = ref.split(".", 1)[1].replace("_", " ").capitalize()
            labels[ref] = (
                st.text_input(
                    f"Libellé de « {ref} »",
                    value=default_label,
                    key=f"db_lbl_{kp}_{ref}",
                ).strip()
                or default_label
            )

    # ---- 4. Module, écran, rôles, URL -----------------------------------------
    with st.container(border=True):
        st.subheader("4. Module, écran, rôles et URL")
        st.caption(
            "Pour chaque information : une valeur fixe, ou une colonne de la base "
            "(valeur propre à chaque ligne)."
        )
        h1, h2 = st.columns(2)
        with h1:
            module = _source_picker(
                "Module", "module", refs, kp, placeholder="Ex. : Ventes"
            )
            roles = _source_picker(
                "Rôles autorisés", "roles", refs, kp, placeholder="all, ou vendeur,pdg"
            )
        with h2:
            ecran = _source_picker("Écran", "ecran", refs, kp, default=table)
            url = _source_picker("URL", "url", refs, kp, placeholder="Facultatif")

    # ---- 5. Ingestion ---------------------------------------------------------
    mapping = {
        "base": {"schema": schema, "table": table},
        "joins": joins,
        "module": module,
        "ecran": ecran,
        "roles": roles,
        "url": url,
        "body": [{"column": r, "label": labels[r]} for r in body_refs],
    }
    roles_ok = bool(roles.get("column") or str(roles.get("value", "")).strip())

    with st.container(border=True):
        st.subheader("5. Ingestion")
        with st.expander("Correspondance envoyée à l'API (JSON)"):
            st.json(mapping)
        if not body_refs:
            st.info("Choisissez au moins une colonne à indexer.")
        elif not roles_ok:
            st.info("Indiquez les rôles autorisés (valeur fixe ou colonne).")

        if st.button(
            "Lancer l'ingestion", type="primary", disabled=not (body_refs and roles_ok)
        ):
            with st.spinner("Lecture des données, formatage et indexation…"):
                ok, res = _db_api(
                    "POST",
                    "/ingest/db/ingest",
                    data={
                        "tenant_id": tenant_id,
                        "connection_id": conn_id,
                        "mapping": json.dumps(mapping),
                    },
                )

            if not ok:
                st.error(res)
            else:
                ing = res.get("ingestion", {})
                if ing.get("status") == "indexed":
                    st.success(
                        f"{res['blocks_ingested']} ligne(s) indexée(s) "
                        f"sur {res['rows_read']} lue(s)."
                    )
                    if res.get("rows_skipped"):
                        st.warning(
                            f"{res['rows_skipped']} ligne(s) ignorée(s) : "
                            "colonnes vides ou rôles manquants."
                        )
                else:
                    st.warning("L'ingestion demande une vérification :")
                    for p in ing.get("problems", []):
                        st.write(f"- {p}")
                st.json(res, expanded=False)
# ==============================================================================
# 5. COUCHE SÉMANTIQUE LANCEDB
# ==============================================================================
elif menu == "Couche sémantique LanceDB":
    page_header(
        "Couche sémantique LanceDB",
        "Chargez les modèles YAML Cube de votre entreprise. L’API vérifie la connexion "
        "analytique, installe les modèles dans Cube et les indexe dans LanceDB.",
    )

    st.caption(f"Entreprise / tenant : **{tenant_id}**")
    connection_ok, connection_result = _db_api(
        "GET",
        "/ingest/semantic-models/connection",
        params={"tenant_id": tenant_id},
    )
    if not connection_ok:
        st.error(connection_result)
        st.stop()
    connection = connection_result.get("connection")
    if not connection:
        st.warning(
            "Aucune connexion n’est enregistrée pour cette entreprise. "
            "Configurez d’abord la base analytique depuis le menu de votre profil."
        )
        st.stop()

    st.subheader("Connexion analytique chargée")
    if st.button("Envoyer la connexion à Cube", key="cube_connect_btn"):
        with st.spinner("Envoi de la connexion à Cube et vérification…"):
            ok, res = _db_api(
                "POST",
                "/ingest/semantic-models/cube-connect",
            )
            if not ok:
                st.error(res)
            else:
                if res.get("db_checked"):
                    st.success(
                        f"Cube est connecté à la base ({res['cubes']} cube(s) chargé(s))."
                    )
                if res.get("dev_ui_ready"):
                    st.info(
                        "L'interface Cube utilise maintenant la connexion de cette entreprise. "
                        "Ouvrez localhost:4000 → Data Model, rechargez la page, sélectionnez "
                        "vos tables puis cliquez sur « Generate Data Model »."
                    )
                else:
                    st.warning(
                        "Connexion envoyée à Cube, mais l'interface de développement n'est pas disponible."
                    )
    with st.form("semantic_models_index_form"):

        st.text_input("Nom de la connexion", value=connection["name"], disabled=True)
        st.text_input(
            "Type de base", value=connection["db_type"].upper(), disabled=True
        )
        c1, c2 = st.columns(2)
        c1.text_input("Hôte", value=connection["host"], disabled=True)
        c2.number_input("Port", value=int(connection["port"]), disabled=True)
        st.text_input("Base de données", value=connection["database"], disabled=True)
        st.text_input("Utilisateur", value=connection["username"], disabled=True)
        st.caption(
            "Le mot de passe reste chiffré côté API; il n’est pas renvoyé à l’interface."
        )

        submitted = st.form_submit_button(
            "Indexer les données sémantique dans LanceDB",
            type="primary",
            width="stretch",
        )

    if submitted:
        tenant_id = tenant_id.lower()
        semantic_data_folder = Path(f"./app/core/{tenant_id}/model/cubes")
        print("[semantic_data_folder]", semantic_data_folder)
        dossier_vide_ou_inexistant = (
            not semantic_data_folder.exists()
            or next(semantic_data_folder.iterdir(), None) is None
        )
        if dossier_vide_ou_inexistant:
            st.error(
                "Rassurez vous que la connexion à votre base de données est bien passé, "
                "que le test a réussi et que les fichiers sémantiques ont été générés depuis l'interface de cube"
            )
        else:
            semantic_files = [
                f
                for f in os.listdir(semantic_data_folder)
                if os.path.isfile(os.path.join(semantic_data_folder, f))
                and f.lower().endswith((".yaml", ".yml"))
            ]
            print("semantic_files", semantic_files)
            if not semantic_files:
                st.error(
                    "Les fichiers sémantiques n'ont pas été générés via l'interface de cubes (localhost:40000"
                )
                print(
                    "Les fichiers sémantiques n'ont pas été générés via l'interface de cubes (localhost:40000"
                )
            else:
                print("fichiers existants et valides dans le dossier de l'entreprise")
                try:
                    with st.spinner("Indexation des fichiers dans LanceDB…"):
                        print("indexation des fichiers")

                        # 1. On prépare proprement les fichiers pour l'envoi multipart
                        multipart_files = []
                        for filename in semantic_files:
                            file_path = os.path.join(semantic_data_folder, filename)
                            # On ouvre chaque fichier en mode lecture binaire
                            f = open(file_path, "rb")  # noqa: SIM115
                            # Le format attendu par requests est ('files', (filename, file_object, content_type))
                            multipart_files.append(
                                ("files", (filename, f, "application/x-yaml"))
                            )

                        # 2. On envoie la requête avec les fichiers et le tenant_id en paramètre de données (data)
                        response = requests.post(
                            f"{FASTAPI_URL}/ingest/semantic-models/index",
                            files=multipart_files,
                            data={"tenant_id": tenant_id},
                            headers=_auth_headers(),
                            timeout=(10, 3600),
                        )

                        # 3. On prend soin de fermer les fichiers ouverts après la requête
                        for _, (_, file_obj, _) in multipart_files:
                            file_obj.close()

                        if response.ok:
                            print("response ok")
                            result = response.json()
                            st.success(
                                f"{result.get('file_indexed', len(semantic_files))} fichiers sémantiques indexé(s) pour "
                                f"l’entreprise {tenant_id}."
                            )
                            st.json(result, expanded=False)
                        else:
                            try:
                                detail = response.json().get("detail", response.text)
                            except ValueError:
                                detail = response.text
                            st.error(
                                f"Échec de l’indexation ({response.status_code}) : {detail}"
                            )
                except requests.RequestException as exc:
                    st.error(f"Impossible de joindre l’API d’indexation : {exc}")

    # ==============================================================================
    # AJOUT : Section pour uploader le fichier de description de la BD (Glossaire)
    # ==============================================================================
    st.divider()
    st.subheader("Dictionnaire et description de la base de données")
    st.caption(
        "Téléversez votre fichier descriptif (Markdown, Texte, PDF, etc.) recensant les tables "
        "et les descriptions des colonnes (ex: rbaseqty_entry, yt_year). "
        "Il sera indexé **brut sans reformatage IA** pour guider précisément l'assistant."
    )

    with st.form("db_glossary_form"):
        desc_file = st.file_uploader(
            "Fichier de description (ex: dictionnaire.md ou .txt)",
            type=["md", "txt", "pdf", "csv"],
            key="db_glossary_uploader",
        )
        desc_submitted = st.form_submit_button(
            "Indexer la description de la BD",
            type="primary",
            use_container_width=True,
        )

    if desc_submitted:
        if not desc_file:
            st.error("Veuillez sélectionner un fichier de description.")
        else:
            with st.spinner("Indexation brute du fichier de description dans LanceDB…"):
                try:
                    files = {
                        "file": (
                            desc_file.name,
                            desc_file.getvalue(),
                            desc_file.type or "application/octet-stream",
                        )
                    }
                    # On force explicitement format_text à False pour sauter le formattage IA
                    data_payload = {
                        "tenant_id": tenant_id,
                        "format_text": "false",
                    }

                    response = requests.post(
                        f"{FASTAPI_URL}/ingest/upload",
                        files=files,
                        data=data_payload,
                        headers=_auth_headers(),
                        timeout=(10, 3600),
                    )

                    if response.status_code == 202:
                        res_data = response.json()
                        st.success(
                            f"Fichier de description **{res_data.get('filename')}** indexé avec succès "
                            f"pour le tenant **{tenant_id}** !"
                        )
                        st.json(res_data, expanded=False)
                    else:
                        try:
                            err_detail = response.json().get("detail", response.text)
                        except ValueError:
                            err_detail = response.text
                        st.error(
                            f"Échec de l'indexation ({response.status_code}) : {err_detail}"
                        )
                except requests.RequestException as exc:
                    st.error(f"Impossible de joindre l'API : {exc}")
