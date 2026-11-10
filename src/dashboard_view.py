"""Rendu du dashboard : cartes animées, thèmes, types de graphiques, filtres, titres éditables."""
import html
import re

import pandas as pd
import plotly.express as px
import streamlit as st

PALETTES = {
    "Océan": ["#2563EB", "#38BDF8", "#0EA5E9", "#1D4ED8", "#7DD3FC", "#1E40AF"],
    "Forêt": ["#15803D", "#4ADE80", "#16A34A", "#86EFAC", "#166534", "#A3E635"],
    "Coucher de soleil": ["#EA580C", "#F59E0B", "#EF4444", "#FB923C", "#DC2626", "#FCD34D"],
    "Violet": ["#7C3AED", "#A78BFA", "#C026D3", "#6D28D9", "#E879F9", "#4C1D95"],
    "Ardoise": ["#1F2A37", "#52606D", "#9AA5B1", "#334155", "#64748B", "#CBD5E1"],
    "Contraste": px.colors.qualitative.Bold,
    "Pastel": px.colors.qualitative.Pastel,
}
INHERIT = "↳ Thème du dashboard"
CHART_TYPES = {
    "vertical_bar": "Barres verticales",
    "horizontal_bar": "Barres horizontales",
    "line": "Courbe",
    "area": "Aire",
    "donut": "Donut",
    "pie": "Camembert",
    "table": "Tableau",
}
GRANULARITIES = ("second", "minute", "hour", "day", "week", "month", "quarter", "year")


# ----------------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------------
def _label(col: str) -> str:
    parts = col.split(".")
    name = parts[1] if len(parts) >= 2 else col
    return name.replace("_", " ").capitalize()


def fmt_num(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:.2f} Md".replace(".", ",")
    if a >= 1e6:
        return f"{x / 1e6:.2f} M".replace(".", ",")
    s = f"{x:,.0f}" if (float(x).is_integer() or a >= 1000) else f"{x:,.2f}"
    return s.replace(",", " ").replace(".", ",")


def default_type(comp_type: str) -> str:
    t = (comp_type or "").lower()
    for key, val in (("metric", "kpi"), ("kpi", "kpi"), ("line", "line"), ("area", "area"),
                     ("donut", "donut"), ("pie", "donut"), ("bar", "vertical_bar")):
        if key in t:
            return val
    return "table"


def prepare(res: dict):
    """Données Cube -> (df, mesures, dimensions, colonne_temps). Cube renvoie les nombres en texte."""
    df = pd.DataFrame(res.get("data") or [])
    if df.empty:
        return df, [], [], None
    q = res.get("cube_query") or res.get("query") or {}

    # Colonne temporelle (clé "cube.dim" ou "cube.dim.month")
    time_base, time_col = None, None
    for td in q.get("timeDimensions", []) or []:
        cands = [c for c in df.columns if c == td.get("dimension") or c.startswith(f"{td.get('dimension')}.")]
        if cands:
            time_base, time_col = td["dimension"], max(cands, key=len)
            break
    if time_col is None:  # inférence si la requête n'est pas transmise
        for c in df.columns:
            if c.split(".")[-1] in GRANULARITIES or df[c].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").all():
                time_col, time_base = c, c.rsplit(".", 1)[0] if c.split(".")[-1] in GRANULARITIES else c
                break

    def is_time(c):
        return time_base and (c == time_base or c.startswith(time_base + "."))

    measures = [m for m in q.get("measures", []) if m in df.columns]
    if not measures:
        measures = [c for c in df.columns if not is_time(c) and c != "query"
                    and pd.to_numeric(df[c], errors="coerce").notna().all()]
    for m in measures:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    dims = [c for c in df.columns if c not in measures and not is_time(c) and c != "query"]

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
    return df, measures, dims, time_col


def _tick_format(time_col: str) -> str:
    g = time_col.split(".")[-1]
    return {"year": "%Y", "day": "%d %b %Y", "week": "%d %b %Y", "quarter": "%b %Y"}.get(g, "%b %Y")


# ----------------------------------------------------------------------------
# Style + animation d'apparition (une carte après l'autre)
# ----------------------------------------------------------------------------
def inject_css(n_cards: int, run: int) -> None:
    # Le nom de l'animation alterne A/B à chaque "run" : le changement de nom relance l'animation.
    name = "dashAppearA" if run % 2 == 0 else "dashAppearB"
    delays = "".join(
        f'[class*="st-key-dcard{i}_"]{{animation-delay:{0.18 * i:.2f}s;}}' for i in range(n_cards)
    )
    kf = "from{opacity:0;transform:translateY(26px) scale(.97);}to{opacity:1;transform:none;}"
    st.markdown(
        f"""<style>
@keyframes dashAppearA {{{kf}}}
@keyframes dashAppearB {{{kf}}}
[class*="st-key-dcard"] {{
    animation: {name} .65s cubic-bezier(.2,.7,.2,1) both;
    border-radius: 12px; box-shadow: 0 1px 4px rgba(16,24,40,.08);
}}
{delays}
.kpi-label {{ font-size:.85rem; color:#52606D; margin:0 0 .15rem; }}
.kpi-value {{ font-size:2.3rem; font-weight:700; line-height:1.1; }}
@media (prefers-reduced-motion: reduce) {{ [class*="st-key-dcard"] {{ animation:none; }} }}
</style>""",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def _figure(agg, ctype, measure, x_col, color_col, time_col, colors, cat_order, show_labels):
    common = dict(color_discrete_sequence=colors, labels={c: _label(c) for c in agg.columns})
    horizontal = ctype == "horizontal_bar"

    if ctype in ("pie", "donut"):
        fig = px.pie(agg, names=x_col, values=measure, hole=0.5 if ctype == "donut" else 0, **common)
        fig.update_traces(textposition="inside", textinfo="percent")
    elif ctype in ("line", "area"):
        fn = px.line if ctype == "line" else px.area
        fig = fn(agg, x=x_col, y=measure, color=color_col, **common,
                 **({"markers": True} if ctype == "line" else {}))
        if show_labels and ctype == "line":
            fig.update_traces(mode="lines+markers+text", texttemplate="%{y:.3s}", textposition="top center")
    else:
        fig = px.bar(agg, x=measure if horizontal else x_col, y=x_col if horizontal else measure,
                     color=color_col, orientation="h" if horizontal else "v", barmode="group",
                     category_orders={x_col: cat_order} if cat_order else None,
                     text_auto=".3s" if show_labels else False, **common)
        if horizontal:
            fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        template="plotly_white", height=340, margin=dict(l=8, r=8, t=8, b=8),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.22, title=None), separators=", ",
    )
    if ctype not in ("pie", "donut"):
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        if time_col and x_col == time_col:
            fig.update_xaxes(tickformat=_tick_format(time_col))
    return fig


# ----------------------------------------------------------------------------
# Une carte
# ----------------------------------------------------------------------------
def render_card(res: dict, idx: int, pos: int, run: int, global_palette: str) -> None:
    ss = st.session_state
    cid = f"{run}_{idx}"
    k = lambda name: f"w{cid}_{name}"  # noqa: E731
    default_title = res.get("component_title") or "Sans titre"
    title = ss.get(k("title")) or default_title          # lu avant le widget : toujours à jour
    kind = default_type(res.get("component_type", ""))

    df, measures, dims, time_col = prepare(res)
    cat_col = dims[0] if dims else None
    x_col = time_col or cat_col
    color_col = cat_col if time_col else (dims[1] if len(dims) > 1 else None)
    is_kpi = kind == "kpi" or (measures and x_col is None)

    with st.container(border=True, key=f"dcard{pos}_{run}"):
        h1, h2 = st.columns([11, 1])
        h1.markdown(f"**{html.escape(title)}**")

        ok = res.get("status") == "success" and not df.empty and bool(measures)
        with h2.popover("⚙️"):
            st.text_input("Titre", value=default_title, key=k("title"))
            pal_name = st.selectbox("🎨 Thème de couleurs", [INHERIT] + list(PALETTES), key=k("pal"))
            colors = PALETTES[global_palette if pal_name == INHERIT else pal_name]
            swatch = "".join(f"<span style='display:inline-block;width:22px;height:12px;background:{c};"
                             f"margin-right:3px;border-radius:3px'></span>" for c in colors[:6])
            st.markdown(swatch, unsafe_allow_html=True)

            ctype, measure, d = kind, (measures[0] if measures else None), df
            top_n, sort_mode, show_labels = 8, "Plus grands", False
            if ok and not is_kpi:
                keys = list(CHART_TYPES)
                ctype = st.selectbox("Type de graphique", keys, index=keys.index(kind) if kind in keys else 0,
                                     format_func=CHART_TYPES.get, key=k("type"))
                if len(measures) > 1:
                    measure = st.selectbox("Mesure", measures, format_func=_label, key=k("m"))
                if time_col and df[time_col].nunique() > 1:
                    lo, hi = df[time_col].min().to_pydatetime(), df[time_col].max().to_pydatetime()
                    p0, p1 = st.slider("Période", lo, hi, (lo, hi), format="DD/MM/YYYY", key=k("dates"))
                    d = d[(d[time_col] >= p0) & (d[time_col] <= p1)]
                if cat_col:
                    picked = st.multiselect("Filtrer les éléments (vide = tous)",
                                            sorted(df[cat_col].astype(str).unique()), key=k("cats"))
                    if picked:
                        d = d[d[cat_col].astype(str).isin(picked)]
                    n_u = d[cat_col].nunique()
                    if n_u > 1:
                        top_n = st.slider("Éléments affichés", 1, n_u, min(n_u, 8), key=k("topn"))
                        sort_mode = st.radio("Tri", ["Plus grands", "Plus petits"], horizontal=True, key=k("sort"))
                show_labels = st.checkbox("Afficher les valeurs", key=k("labels"))
            if ok:
                fname = re.sub(r"\W+", "_", title) + ".csv"
                st.download_button("⬇️ Données (CSV)", d.to_csv(index=False).encode("utf-8"),
                                   file_name=fname, key=k("csv"))

        # ---------------- Rendu ----------------
        if res.get("status") != "success":
            st.error(f"Erreur : {res.get('error', 'inconnue')}")
            return
        if df.empty or not measures:
            st.info("Aucune donnée pour cette requête (voir le mode Debug pour la requête Cube).")
            return

        if is_kpi:
            val = fmt_num(d[measure].iloc[0])
            st.markdown(
                f"<div style='border-left:5px solid {colors[0]};padding-left:14px'>"
                f"<p class='kpi-label'>{html.escape(_label(measure))}</p>"
                f"<div class='kpi-value'>{val}</div></div>",
                unsafe_allow_html=True,
            )
            return
        if d.empty:
            st.info("Aucune donnée sur cette sélection.")
            return

        if ctype == "table":
            cols = [c for c in (time_col, *dims, *measures) if c]
            t = d[cols].sort_values(measure, ascending=(sort_mode == "Plus petits")).head(50)
            st.dataframe(t.rename(columns={c: _label(c) for c in cols}), hide_index=True,
                         use_container_width=True, height=min(380, 45 + 35 * len(t)))
            return

        group = [c for c in (x_col, color_col) if c]
        agg = d.groupby(group, as_index=False)[measure].sum()
        cat_order = None
        if cat_col:
            totals = agg.groupby(cat_col)[measure].sum().sort_values(ascending=(sort_mode == "Plus petits"))
            keep = list(totals.head(top_n).index)
            if ctype in ("pie", "donut") and x_col == cat_col and len(totals) > top_n:
                other = totals.drop(keep).sum()
                agg = pd.concat([agg[agg[cat_col].isin(keep)],
                                 pd.DataFrame({cat_col: ["Autres"], measure: [other]})], ignore_index=True)
            else:
                agg = agg[agg[cat_col].isin(keep)]
            cat_order = keep
        if time_col and x_col == time_col:
            agg = agg.sort_values(time_col)
            if agg[time_col].nunique() <= 1 and ctype in ("line", "area"):
                ctype = "vertical_bar"
                st.caption("Une seule période dans les données : affichage en barres.")

        fig = _figure(agg, ctype, measure, x_col, color_col, time_col, colors, cat_order, show_labels)
        st.plotly_chart(fig, use_container_width=True, key=k("fig"), config={"displayModeBar": False})


# ----------------------------------------------------------------------------
# Dashboard complet
# ----------------------------------------------------------------------------
def render_dashboard(results: list, dash_json: dict) -> None:
    ss = st.session_state
    ss.setdefault("dash_run", 0)
    ss.setdefault("dash_title", (dash_json or {}).get("dashboard_title") or "Tableau de bord")
    run = ss["dash_run"]

    c1, c2, c3, c4 = st.columns([6, 3, 1, 1], vertical_alignment="center")
    c1.markdown(f"## 📌 {html.escape(ss['dash_title'] or 'Tableau de bord')}")
    global_palette = c2.selectbox("🎨 Thème du dashboard", list(PALETTES), key="dash_palette")
    with c3.popover("✏️"):
        st.text_input("Titre du dashboard", key="dash_title")
    if c4.button("🎞️", help="Rejouer l'animation"):
        ss["dash_run"] += 1
        st.rerun()

    # Ordre d'affichage : KPI -> graphiques -> tableaux
    typed = [(i, r, default_type(r.get("component_type", ""))) for i, r in enumerate(results)]
    kpis = [t for t in typed if t[2] == "kpi"]
    tables = [t for t in typed if t[2] == "table"]
    charts = [t for t in typed if t[2] not in ("kpi", "table")]
    inject_css(len(typed), run)

    pos = 0
    if kpis:
        for col, (i, r, _) in zip(st.columns(len(kpis)), kpis):
            with col:
                render_card(r, i, pos, run, global_palette)
            pos += 1
    for n in range(0, len(charts), 2):
        chunk = charts[n:n + 2]
        for col, (i, r, _) in zip(st.columns(len(chunk)), chunk):
            with col:
                render_card(r, i, pos, run, global_palette)
            pos += 1
    for i, r, _ in tables:
        render_card(r, i, pos, run, global_palette)
        pos += 1