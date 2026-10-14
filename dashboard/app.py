"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Cliente 360°  –  Dashboard Interactivo de Marketing                 ║
║          Red Neuronal Multi-Tarea  |  MIPD – ULPGC                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Secciones:
  1. 👤 Ficha del Cliente 360°        (Operational Marketing)
  2. 🎯 Segmentación por Perfil        (Strategic Marketing)
  3. 🛍️  Recomendación y Catálogo      (Product Marketing)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

# Raíz del proyecto (un nivel arriba del directorio dashboard/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "neural-network" / "artifacts"
PROCESSED_DIR = PROJECT_ROOT / "data_processing" / "data" / "processed"

COLORS = {
    "primary":  "#6C63FF",
    "secondary": "#F76C6C",
    "accent":   "#43E8D8",
    "success":  "#2ECC71",
    "warning":  "#F39C12",
    "dark_bg":  "#0E1117",
    "card_bg":  "#1A1D2E",
    "text":     "#FAFAFA",
    "muted":    "#8B8FA8",
}

SEGMENT_CONFIG = {
    0: {"label": "Champions",       "color": "#6C63FF", "emoji": "🏆",
        "accion": "Campaña VIP exclusiva – early access a nuevos productos y rewards premium."},
    1: {"label": "Leales",          "color": "#2ECC71", "emoji": "💚",
        "accion": "Programa de fidelización – puntos dobles y descuentos en categorías favoritas."},
    2: {"label": "Potenciales",     "color": "#43E8D8", "emoji": "🌱",
        "accion": "Upselling – sugerencias de categorías complementarias con oferta de bienvenida."},
    3: {"label": "En Riesgo",       "color": "#F39C12", "emoji": "⚠️",
        "accion": "Re-engagement por email/push – cupón de descuento con caducidad corta."},
    4: {"label": "Hibernando",      "color": "#F76C6C", "emoji": "😴",
        "accion": "Campaña de reactivación – win-back con descuento agresivo (30-40%)."},
    5: {"label": "Nuevos",          "color": "#E91E63", "emoji": "✨",
        "accion": "Onboarding – guía personalizada + descuento en primera recompra."},
    6: {"label": "Ocasionales",     "color": "#9E9E9E", "emoji": "🛒",
        "accion": "Cross-selling – recomendaciones de productos de alto margen en categorías vistas."},
    7: {"label": "Alta Frecuencia", "color": "#FF9800", "emoji": "⚡",
        "accion": "Bundle offers – packs temáticos con descuento por volumen."},
}

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS CON CACHÉ
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner="Cargando predicciones CLV…")
def cargar_predictions_clv() -> pd.DataFrame | None:
    ruta = ARTIFACTS_DIR / "predictions_clv.csv"
    if not ruta.exists():
        return None
    df = pd.read_csv(ruta)
    df["clv_original_pred"] = df["clv_original_pred"].clip(lower=0)
    return df


@st.cache_data(ttl=600, show_spinner="Cargando recomendaciones…")
def cargar_predictions_rec() -> pd.DataFrame | None:
    ruta = ARTIFACTS_DIR / "predictions_rec.csv"
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)


@st.cache_data(ttl=600, show_spinner="Cargando datos de clientes…")
def cargar_clientes() -> pd.DataFrame | None:
    ruta = PROCESSED_DIR / "clientes_test.parquet"
    if not ruta.exists():
        return None
    try:
        return pd.read_parquet(ruta)
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner="Cargando embeddings…")
def cargar_embeddings() -> np.ndarray | None:
    ruta = ARTIFACTS_DIR / "customer_embeddings.npy"
    if not ruta.exists():
        return None
    return np.load(str(ruta))


@st.cache_data(ttl=600, show_spinner="Cargando mapeo de clientes…")
def cargar_customer_idx_to_id() -> dict | None:
    ruta = ARTIFACTS_DIR / "customer_idx_to_id.json"
    if not ruta.exists():
        return None
    import json
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=600, show_spinner="Cargando mapeo de productos…")
def cargar_product_idx_to_id() -> dict | None:
    ruta = ARTIFACTS_DIR / "product_idx_to_id.json"
    if not ruta.exists():
        return None
    import json
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Calculando segmentos (K-Means sobre embeddings)…")
def calcular_segmentos(n_clusters: int = 6) -> pd.DataFrame | None:
    """KMeans sobre PCA de customer_embeddings – ejecutado una sola vez."""
    emb = cargar_embeddings()
    if emb is None:
        return None
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(emb)

        pca_full = PCA(n_components=min(16, emb.shape[1]), random_state=42)
        emb_pca  = pca_full.fit_transform(emb_scaled)
        km       = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels   = km.fit_predict(emb_pca)

        pca2d   = PCA(n_components=2, random_state=42)
        coords  = pca2d.fit_transform(emb_scaled[:, :min(32, emb.shape[1])])

        return pd.DataFrame({
            "customer_idx": np.arange(len(labels)),
            "segment":      labels,
            "pca_x":        coords[:, 0],
            "pca_y":        coords[:, 1],
        })
    except ImportError:
        st.error("Instala scikit-learn: pip install scikit-learn")
        return None


@st.cache_data(show_spinner="Enriqueciendo tabla de clientes…")
def construir_tabla_clientes_enriquecida() -> pd.DataFrame | None:
    pred_clv  = cargar_predictions_clv()
    clientes  = cargar_clientes()
    seg       = calcular_segmentos()

    if pred_clv is None:
        return None

    df = pred_clv.copy()

    if clientes is not None:
        cols_merge = [c for c in clientes.columns if c not in df.columns or c == "customer_idx"]
        df = df.merge(clientes[cols_merge], on="customer_idx", how="left")

    if seg is not None:
        df = df.merge(seg[["customer_idx", "segment", "pca_x", "pca_y"]], on="customer_idx", how="left")
        df["segment"] = df["segment"].fillna(-1).astype(int)

    c2id = cargar_customer_idx_to_id()
    if c2id:
        df["customer_unique_id"] = df["customer_idx"].astype(str).map(c2id).fillna(df["customer_idx"].astype(str))
    else:
        df["customer_unique_id"] = df["customer_idx"].astype(str)

    df["segment_label"] = df["segment"].map(
        lambda s: SEGMENT_CONFIG.get(s, {}).get("label", f"Segmento {s}")
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS UI
# ─────────────────────────────────────────────────────────────────────────────

def badge_segmento(seg_id: int) -> str:
    cfg = SEGMENT_CONFIG.get(seg_id, {"label": f"Seg {seg_id}", "color": "#9E9E9E", "emoji": "❓"})
    c = cfg["color"]
    return (
        f"<span style='background:{c}22;color:{c};border:1px solid {c}55;"
        f"padding:3px 10px;border-radius:20px;font-size:0.85rem;font-weight:600'>"
        f"{cfg['emoji']} {cfg['label']}</span>"
    )


def kpi_card(title: str, value: str, delta: str = "", icon: str = "",
             color: str = COLORS["primary"]) -> str:
    cb  = COLORS["card_bg"]
    ct  = COLORS["text"]
    cm  = COLORS["muted"]
    dh  = f"<p style='color:{cm};font-size:0.75rem;margin:4px 0 0'>{delta}</p>" if delta else ""
    return (
        f"<div style='background:{cb};border:1px solid {color}33;border-left:4px solid {color};"
        f"border-radius:12px;padding:18px 20px;margin-bottom:12px'>"
        f"<p style='color:{cm};font-size:0.78rem;margin:0;text-transform:uppercase;"
        f"letter-spacing:0.05em'>{icon} {title}</p>"
        f"<p style='color:{ct};font-size:1.65rem;font-weight:700;margin:6px 0 0'>{value}</p>"
        f"{dh}</div>"
    )


def error_card(msg: str) -> None:
    st.markdown(
        f"<div style='background:#F76C6C15;border:1px solid #F76C6C44;border-radius:12px;"
        f"padding:16px 20px;margin:8px 0;color:#F76C6C'>⚠️ {msg}</div>",
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    pc = COLORS["primary"]
    mc = COLORS["muted"]
    st.markdown(f"<h2 style='color:{pc};margin-bottom:2px'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='color:{mc};margin-top:0;font-size:0.9rem'>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2A2D3E;margin:8px 0 20px'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1: FICHA DEL CLIENTE 360°
# ─────────────────────────────────────────────────────────────────────────────

def seccion_ficha_cliente() -> None:
    section_header(
        "👤 Ficha del Cliente 360°",
        "Perfil individual, CLV predicho, segmento asignado y recomendaciones personalizadas",
    )

    df_cli = construir_tabla_clientes_enriquecida()
    if df_cli is None:
        error_card("No se encontraron predicciones CLV. Ejecuta primero el pipeline de entrenamiento.")
        return

    df_rec = cargar_predictions_rec()
    p2id   = cargar_product_idx_to_id()

    # ── Buscador ──────────────────────────────────────────────────────────── #
    ids_display = df_cli["customer_unique_id"].tolist()
    col_s, col_r = st.columns([4, 1])
    with col_s:
        selected = st.selectbox(
            "🔍 Selecciona un Customer ID",
            options=ids_display, index=0, key="cliente_selector",
            help="ID único de cliente del dataset Olist",
        )
    with col_r:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Aleatorio", key="btn_random", width="stretch"):
            st.session_state["cliente_selector"] = ids_display[np.random.randint(len(ids_display))]
            st.rerun()

    row = df_cli[df_cli["customer_unique_id"] == selected]
    if row.empty:
        error_card(f"Cliente '{selected}' no encontrado.")
        return
    row = row.iloc[0]

    seg_id  = int(row.get("segment", -1))
    seg_cfg = SEGMENT_CONFIG.get(seg_id, {"label": "Desconocido", "color": "#9E9E9E",
                                           "emoji": "❓", "accion": "—"})

    st.markdown(f"<div style='margin:8px 0 16px'>{badge_segmento(seg_id)}</div>", unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────────────── #
    st.markdown("#### 📊 Indicadores Clave")
    c1, c2, c3, c4 = st.columns(4)

    clv_pred   = float(row.get("clv_original_pred", 0))
    clv_real   = float(row.get("clv_original_real", row.get("clv_futuro", 0)))
    historico  = float(row.get("importe_total", 0))
    frecuencia = int(row.get("frecuencia", 0))
    recompra   = float(row.get("recompra", 0))
    recencia   = int(row.get("recencia_dias", 0))
    antiguedad = int(row.get("antiguedad_dias", 0))

    with c1:
        st.markdown(kpi_card("CLV Futuro Predicho", f"R$ {clv_pred:,.2f}",
                             f"Real observado: R$ {clv_real:,.2f}", "💰", COLORS["primary"]),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Gasto Histórico", f"R$ {historico:,.2f}",
                             f"{frecuencia} pedido{'s' if frecuencia != 1 else ''}", "🛒", COLORS["accent"]),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("¿Recompró?", "Sí ✓" if recompra > 0 else "No ✗",
                             "Recompra observada en ventana futura", "🔄", COLORS["success"]),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Recencia", f"{recencia} días",
                             f"Antigüedad: {antiguedad} días", "📅", COLORS["warning"]),
                    unsafe_allow_html=True)

    # ── Gauge CLV ──────────────────────────────────────────────────────────── #
    st.markdown("#### 🎯 Posición de CLV en la Distribución")
    clv_max  = float(df_cli["clv_original_pred"].quantile(0.99))
    clv_mean = float(df_cli["clv_original_pred"].mean())

    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=clv_pred,
        delta={"reference": clv_mean, "valueformat": ".2f", "prefix": "Media: R$ "},
        number={"prefix": "R$ ", "valueformat": ".2f", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, clv_max], "tickprefix": "R$ "},
            "bar":  {"color": seg_cfg["color"]},
            "bgcolor": COLORS["card_bg"], "bordercolor": "#2A2D3E",
            "steps": [
                {"range": [0, clv_max * 0.25], "color": "#F76C6C22"},
                {"range": [clv_max * 0.25, clv_max * 0.6], "color": "#F39C1222"},
                {"range": [clv_max * 0.6, clv_max], "color": "#2ECC7122"},
            ],
            "threshold": {"line": {"color": COLORS["primary"], "width": 3},
                          "thickness": 0.8, "value": clv_mean},
        },
        title={"text": "CLV Predicho (R$)", "font": {"size": 14, "color": COLORS["muted"]}},
    ))
    fig_g.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=20, r=20, t=40, b=10),
        font={"color": COLORS["text"], "family": "Inter"},
    )
    st.plotly_chart(fig_g, width="stretch")

    # ── Recomendaciones NCF ────────────────────────────────────────────────── #
    st.markdown("#### 🛍️ Recomendaciones Personalizadas (Neural CF Top-K)")
    if df_rec is not None:
        c_idx       = int(row["customer_idx"])
        recs_cli    = df_rec[df_rec["customer_idx"] == c_idx].sort_values("rank")
        if not recs_cli.empty:
            k        = st.slider("Número de recomendaciones", 3, 20, 10, key="k_cliente")
            recs_top = recs_cli.head(k).copy()
            if p2id:
                recs_top["product_id"] = recs_top["product_idx"].astype(str).map(p2id).fillna(
                    recs_top["product_idx"].astype(str)
                )
            else:
                recs_top["product_id"] = recs_top["product_idx"].astype(str)

            s_min = recs_top["score"].min(); s_max = recs_top["score"].max()
            recs_top["relevancia"] = ((recs_top["score"] - s_min) / max(s_max - s_min, 1e-9) * 100).round(1)

            fig_r = go.Figure(go.Bar(
                x=recs_top["relevancia"].values[::-1],
                y=[
                    f"#{r} · {str(pid)[:18]}…" if len(str(pid)) > 18 else f"#{r} · {pid}"
                    for r, pid in zip(recs_top["rank"].values[::-1], recs_top["product_id"].values[::-1])
                ],
                orientation="h",
                marker=dict(color=recs_top["relevancia"].values[::-1],
                            colorscale=[[0, "#6C63FF33"], [1, COLORS["primary"]]]),
                text=[f"{v:.1f}%" for v in recs_top["relevancia"].values[::-1]],
                textposition="outside",
            ))
            fig_r.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Relevancia Normalizada (%)", showgrid=True, gridcolor="#2A2D3E"),
                yaxis=dict(showgrid=False),
                height=max(300, k * 34),
                margin=dict(l=10, r=70, t=10, b=30),
                font={"color": COLORS["text"], "family": "Inter"},
            )
            st.plotly_chart(fig_r, width="stretch")
        else:
            st.info(f"ℹ️ El cliente idx={c_idx} no tiene recomendaciones exportadas (cold-start o fuera del split de evaluación).")
    else:
        error_card("predictions_rec.csv no encontrado en artifacts/.")

    # ── Perfil histórico ───────────────────────────────────────────────────── #
    st.markdown("#### 📋 Perfil de Comportamiento Histórico")
    feats = {
        "Frecuencia":         f"{frecuencia} pedidos",
        "Importe total":      f"R$ {historico:,.2f}",
        "Ticket medio":       f"R$ {float(row.get('ticket_medio', 0)):,.2f}",
        "Cuotas medias":      f"{float(row.get('cuotas_media', 0)):.1f}",
        "Recencia":           f"{recencia} días",
        "Antigüedad":         f"{antiguedad} días",
        "Días entre pedidos": f"{float(row.get('dias_entre_pedidos', 0)):.0f} días",
        "N° Categorías":      f"{int(row.get('n_categorias', 0))}",
    }
    cols = st.columns(4)
    for i, (label, val) in enumerate(feats.items()):
        with cols[i % 4]:
            st.metric(label=label, value=val)

    # Acción de marketing
    color_s = seg_cfg["color"]
    st.markdown(
        f"<div style='background:{color_s}15;border:1px solid {color_s}44;border-radius:12px;"
        f"padding:16px 20px;margin-top:16px'>"
        f"<p style='color:{COLORS['muted']};font-size:0.78rem;margin:0;text-transform:uppercase;"
        f"letter-spacing:0.05em'>💡 Acción de Marketing – {seg_cfg['emoji']} {seg_cfg['label']}</p>"
        f"<p style='color:{COLORS['text']};font-size:1rem;margin:8px 0 0;font-weight:500'>"
        f"{seg_cfg['accion']}</p></div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2: SEGMENTACIÓN Y ESTRATEGIAS POR PERFIL
# ─────────────────────────────────────────────────────────────────────────────

def seccion_segmentacion() -> None:
    section_header(
        "🎯 Segmentación y Estrategias por Perfil",
        "Clusters de clientes sobre embeddings aprendidos por la red – visualización PCA 2D",
    )

    n_clusters = st.sidebar.slider("Número de segmentos (K)", 2, 8, 6, key="n_clusters_slider")
    df_seg     = calcular_segmentos(n_clusters=n_clusters)
    df_clv     = construir_tabla_clientes_enriquecida()

    if df_seg is None:
        error_card("No se pudieron calcular segmentos. Verifica customer_embeddings.npy y scikit-learn.")
        return

    # Enriquecer con CLV
    if df_clv is not None:
        mc = [c for c in ["customer_idx", "clv_original_pred", "importe_total", "frecuencia",
                           "ticket_medio", "recencia_dias", "n_categorias", "customer_unique_id"]
              if c in df_clv.columns]
        df_viz = df_seg.merge(df_clv[mc].drop_duplicates("customer_idx"), on="customer_idx", how="left")
    else:
        df_viz = df_seg.copy(); df_viz["clv_original_pred"] = np.nan

    df_viz["segment_label"] = df_viz["segment"].map(
        lambda s: f"{SEGMENT_CONFIG.get(s, {}).get('emoji', '')} "
                  f"{SEGMENT_CONFIG.get(s, {}).get('label', f'Seg {s}')}"
    )
    seg_colors = {
        f"{SEGMENT_CONFIG.get(s, {}).get('emoji', '')} "
        f"{SEGMENT_CONFIG.get(s, {}).get('label', f'Seg {s}')}":
        SEGMENT_CONFIG.get(s, {}).get("color", "#9E9E9E")
        for s in range(n_clusters)
    }

    # ── Mapa 2D ───────────────────────────────────────────────────────────── #
    st.markdown("#### 🗺️ Mapa de Embeddings (PCA 2D)")
    sample_sz  = min(8000, len(df_viz))
    df_sample  = df_viz.sample(sample_sz, random_state=42) if len(df_viz) > sample_sz else df_viz
    hover_cols = ["customer_idx"] + (
        ["customer_unique_id"] if "customer_unique_id" in df_sample.columns else []
    ) + (["clv_original_pred"] if "clv_original_pred" in df_sample.columns else [])

    fig_m = px.scatter(
        df_sample, x="pca_x", y="pca_y",
        color="segment_label", color_discrete_map=seg_colors,
        hover_data=hover_cols, opacity=0.65,
        title=f"Distribución de {sample_sz:,} clientes en espacio de embeddings",
        labels={"pca_x": "PC1", "pca_y": "PC2", "segment_label": "Segmento"},
    )
    fig_m.update_traces(marker=dict(size=4))
    fig_m.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=COLORS["card_bg"],
        xaxis=dict(showgrid=True, gridcolor="#2A2D3E", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#2A2D3E", zeroline=False),
        height=500, legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2A2D3E"),
        font={"color": COLORS["text"], "family": "Inter"}, margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_m, width="stretch")

    # ── Profiling ─────────────────────────────────────────────────────────── #
    st.markdown("#### 📊 Profiling Comparativo de Segmentos")
    agg: dict = {"customer_idx": "count"}
    for col in ["clv_original_pred", "importe_total", "frecuencia",
                "ticket_medio", "recencia_dias", "n_categorias"]:
        if col in df_viz.columns:
            agg[col] = "mean"

    df_prof = df_viz.groupby("segment").agg(agg).reset_index()
    df_prof = df_prof.rename(columns={
        "customer_idx": "N° Clientes", "clv_original_pred": "CLV Pred Medio (R$)",
        "importe_total": "Gasto Hist. Medio (R$)", "frecuencia": "Frecuencia Media",
        "ticket_medio": "Ticket Medio (R$)", "recencia_dias": "Recencia (días)",
        "n_categorias": "N° Categorías",
    })
    df_prof["Segmento"] = df_prof["segment"].map(
        lambda s: f"{SEGMENT_CONFIG.get(s, {}).get('emoji', '')} "
                  f"{SEGMENT_CONFIG.get(s, {}).get('label', f'Seg {s}')}"
    )
    disp_prof = df_prof.drop(columns=["segment"]).set_index("Segmento")
    for col in ["CLV Pred Medio (R$)", "Gasto Hist. Medio (R$)", "Ticket Medio (R$)"]:
        if col in disp_prof.columns:
            disp_prof[col] = disp_prof[col].map(lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "—")
    for col in ["Frecuencia Media", "Recencia (días)", "N° Categorías"]:
        if col in disp_prof.columns:
            disp_prof[col] = disp_prof[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    st.dataframe(disp_prof, width="stretch", height=280)

    # ── Gráficos ──────────────────────────────────────────────────────────── #
    col_pie, col_bar = st.columns(2)
    if "segment" not in df_prof.columns:
        df_prof["segment"] = range(len(df_prof))
    c_pie = [SEGMENT_CONFIG.get(int(s), {}).get("color", "#9E9E9E") for s in df_viz["segment"].unique()]
    label_pie = [
        f"{SEGMENT_CONFIG.get(int(s), {}).get('emoji', '')} "
        f"{SEGMENT_CONFIG.get(int(s), {}).get('label', f'Seg {s}')}"
        for s in df_viz["segment"].unique()
    ]
    cnt_pie = [len(df_viz[df_viz["segment"] == s]) for s in df_viz["segment"].unique()]

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=label_pie, values=cnt_pie,
            marker=dict(colors=c_pie, line=dict(color=COLORS["dark_bg"], width=2)),
            textinfo="label+percent", hole=0.45,
        ))
        fig_pie.update_layout(
            title="Distribución de Clientes por Segmento",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": COLORS["text"], "family": "Inter"},
            height=360, showlegend=False, margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_pie, width="stretch")

    with col_bar:
        if "clv_original_pred" in df_viz.columns:
            segs_sorted = sorted(df_viz["segment"].unique())
            clv_by_seg  = [df_viz[df_viz["segment"] == s]["clv_original_pred"].mean() for s in segs_sorted]
            labels_b    = [
                f"{SEGMENT_CONFIG.get(s, {}).get('emoji', '')} "
                f"{SEGMENT_CONFIG.get(s, {}).get('label', f'Seg {s}')}"
                for s in segs_sorted
            ]
            colors_b = [SEGMENT_CONFIG.get(s, {}).get("color", "#9E9E9E") for s in segs_sorted]
            fig_b = go.Figure(go.Bar(
                x=labels_b, y=clv_by_seg, marker=dict(color=colors_b),
                text=[f"R$ {v:,.0f}" for v in clv_by_seg], textposition="outside",
            ))
            fig_b.update_layout(
                title="CLV Predicho Medio por Segmento",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(title="CLV Medio (R$)", showgrid=True, gridcolor="#2A2D3E"),
                font={"color": COLORS["text"], "family": "Inter"},
                height=360, margin=dict(l=10, r=10, t=50, b=60),
            )
            st.plotly_chart(fig_b, width="stretch")

    # ── Acciones de marketing ─────────────────────────────────────────────── #
    st.markdown("#### 💡 Acciones de Marketing por Segmento")
    for seg_id in range(n_clusters):
        cfg_s  = SEGMENT_CONFIG.get(seg_id, {})
        n_seg  = len(df_viz[df_viz["segment"] == seg_id])
        label  = cfg_s.get("label", f"Segmento {seg_id}")
        emoji  = cfg_s.get("emoji", "🔘")
        color  = cfg_s.get("color", "#9E9E9E")
        accion = cfg_s.get("accion", "Definir estrategia específica.")

        with st.expander(f"{emoji} Segmento {seg_id} – {label}  ({n_seg:,} clientes)", expanded=False):
            st.markdown(
                f"<div style='background:{color}15;border-left:4px solid {color};"
                f"border-radius:0 8px 8px 0;padding:12px 16px'>"
                f"<p style='color:{COLORS['text']};margin:0'><strong>🎯 Acción sugerida:</strong> {accion}</p>"
                f"</div>", unsafe_allow_html=True,
            )
            seg_d = df_viz[df_viz["segment"] == seg_id]
            ca, cb, cc = st.columns(3)
            with ca: st.metric("Clientes", f"{n_seg:,}")
            with cb:
                if "clv_original_pred" in seg_d.columns:
                    st.metric("CLV Medio", f"R$ {seg_d['clv_original_pred'].mean():,.2f}")
            with cc:
                if "frecuencia" in seg_d.columns:
                    st.metric("Frecuencia Media", f"{seg_d['frecuencia'].mean():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3: RECOMENDACIÓN Y ANÁLISIS DE CATÁLOGO
# ─────────────────────────────────────────────────────────────────────────────

def seccion_recomendacion() -> None:
    section_header(
        "🛍️ Recomendación y Análisis de Catálogo",
        "Probador interactivo del recomendador NCF y métricas de cobertura del catálogo Olist",
    )

    df_rec = cargar_predictions_rec()
    df_cli = construir_tabla_clientes_enriquecida()
    p2id   = cargar_product_idx_to_id()

    if df_rec is None:
        error_card("predictions_rec.csv no encontrado en artifacts/. Ejecuta el pipeline primero.")
        return

    # ── Probador ──────────────────────────────────────────────────────────── #
    st.markdown("#### 🔮 Simulador de Recomendaciones")
    c_disponibles = sorted(df_rec["customer_idx"].unique())
    ca, cb = st.columns([3, 1])
    with ca:
        idx_sel = st.selectbox(
            "Customer Index (con recomendaciones generadas)",
            options=c_disponibles[:500], key="rec_customer_idx",
            format_func=lambda x: (
                f"idx={x} | {df_cli[df_cli['customer_idx']==x]['customer_unique_id'].values[0][:24]}"
                if df_cli is not None and not df_cli[df_cli["customer_idx"]==x].empty
                else f"idx={x}"
            ),
        )
    with cb:
        k_rec = st.number_input("Top K", 1, 20, 10, key="k_recomendador")

    recs = df_rec[df_rec["customer_idx"] == idx_sel].sort_values("rank").head(k_rec).copy()
    if recs.empty:
        st.info("ℹ️ Sin recomendaciones para este cliente en el conjunto exportado.")
    else:
        if p2id:
            recs["product_id"] = recs["product_idx"].astype(str).map(p2id).fillna(
                recs["product_idx"].astype(str)
            )
        else:
            recs["product_id"] = recs["product_idx"].astype(str)
        disp = recs[["rank", "product_id", "product_idx", "score"]].rename(columns={
            "rank": "Rank", "product_id": "Product ID", "product_idx": "Idx", "score": "Score (logit)"
        })
        st.dataframe(disp, width="stretch", hide_index=True)

    # ── Cobertura del catálogo ────────────────────────────────────────────── #
    st.markdown("#### 📈 Métricas de Catálogo y Cobertura")
    N_PROD   = 32216  # config.yaml → modelo.n_productos
    n_unicos = df_rec["product_idx"].nunique()
    n_c_recs = df_rec["customer_idx"].nunique()
    cob_pct  = n_unicos / N_PROD * 100

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown(kpi_card("Cobertura del Catálogo", f"{cob_pct:.1f}%",
                             f"{n_unicos:,} / {N_PROD:,} productos", "📦", COLORS["primary"]),
                    unsafe_allow_html=True)
    with cc2:
        st.markdown(kpi_card("Clientes con Recs", f"{n_c_recs:,}",
                             "En conjunto de test", "👥", COLORS["accent"]),
                    unsafe_allow_html=True)
    with cc3:
        st.markdown(kpi_card("Score Medio (logit)", f"{df_rec['score'].mean():.4f}",
                             "Logit promedio", "📊", COLORS["success"]),
                    unsafe_allow_html=True)

    # Frecuencias
    prod_freq = df_rec["product_idx"].value_counts().reset_index()
    prod_freq.columns = ["product_idx", "n_rec"]

    col_h, col_t = st.columns(2)
    with col_h:
        fig_h = px.histogram(prod_freq, x="n_rec", nbins=60,
                             title="Distribución de Frecuencia de Recomendación (Long Tail)",
                             color_discrete_sequence=[COLORS["primary"]])
        fig_h.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=COLORS["card_bg"],
            xaxis_title="Nº de veces recomendado", yaxis_title="Nº de productos",
            xaxis=dict(showgrid=True, gridcolor="#2A2D3E"),
            yaxis=dict(showgrid=True, gridcolor="#2A2D3E"),
            font={"color": COLORS["text"], "family": "Inter"},
            height=350, margin=dict(l=10, r=10, t=50, b=30),
        )
        st.plotly_chart(fig_h, width="stretch")

    with col_t:
        top20 = prod_freq.head(20).copy()
        top20["label"] = (top20["product_idx"].astype(str).map(p2id)
                          if p2id else top20["product_idx"].astype(str))
        top20["label"] = top20["label"].apply(lambda x: str(x)[:16] + "…" if len(str(x)) > 16 else str(x))
        fig_t = px.bar(top20.sort_values("n_rec"), x="n_rec", y="label", orientation="h",
                       title="Top 20 Productos más Recomendados",
                       color="n_rec", color_continuous_scale=[[0, "#6C63FF44"], [1, COLORS["primary"]]])
        fig_t.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=COLORS["card_bg"],
            xaxis=dict(showgrid=True, gridcolor="#2A2D3E"), yaxis=dict(showgrid=False),
            coloraxis_showscale=False, font={"color": COLORS["text"], "family": "Inter"},
            height=350, margin=dict(l=10, r=10, t=50, b=20),
        )
        st.plotly_chart(fig_t, width="stretch")

    # Gini
    fv = np.sort(prod_freq["n_rec"].values); n = len(fv); tot = fv.sum()
    gini = (2 * np.sum(np.arange(1, n+1) * fv) / (n * tot) - (n+1)/n) if tot > 0 else 0.0
    nivel = ("Alta concentración (problema de popularidad)" if gini > 0.7
             else "Distribución moderada" if gini > 0.4
             else "Buena diversidad en recomendaciones")
    st.info(f"📐 **Índice de Gini de recomendaciones:** `{gini:.4f}` — {nivel}")


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Cliente 360° | MIPD Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pc  = COLORS["primary"]
    cbc = COLORS["card_bg"]
    mc  = COLORS["muted"]

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .main .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0E1117 0%, #1A1D2E 100%);
            border-right: 1px solid #2A2D3E;
        }}
        .stMetric {{
            background: {cbc}; border: 1px solid #2A2D3E;
            border-radius: 12px; padding: 12px 16px;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px; background-color: {cbc}; border-radius: 10px; padding: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px; color: {mc}; font-weight: 500;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {pc}; color: white !important;
        }}
        div[data-testid="stExpander"] {{
            background: {cbc}; border: 1px solid #2A2D3E; border-radius: 12px;
        }}
        .stButton > button {{
            background: linear-gradient(135deg, {pc}, #8B5CF6);
            color: white; border: none; border-radius: 8px; font-weight: 600;
            transition: all 0.2s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px); box-shadow: 0 4px 15px {pc}55;
        }}
    </style>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────────────── #
    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center;padding:20px 0 24px'>"
            f"<div style='font-size:3rem'>🧠</div>"
            f"<h2 style='color:{pc};margin:8px 0 4px;font-size:1.1rem'>Cliente 360°</h2>"
            f"<p style='color:{mc};font-size:0.78rem;margin:0'>MultiTaskNet · MIPD · ULPGC</p></div>"
            f"<hr style='border-color:#2A2D3E;margin:0 0 20px'>",
            unsafe_allow_html=True,
        )

        seccion = st.radio(
            "Navegación",
            options=[
                "👤 Ficha del Cliente 360°",
                "🎯 Segmentación por Perfil",
                "🛍️ Recomendación y Catálogo",
            ],
            key="nav_section", label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#2A2D3E;margin:20px 0'>", unsafe_allow_html=True)

        # Estado de artefactos
        st.markdown(
            f"<p style='color:{mc};font-size:0.75rem;text-transform:uppercase;"
            f"letter-spacing:0.05em;margin-bottom:8px'>Estado de Artefactos</p>",
            unsafe_allow_html=True,
        )
        for nombre, ruta in {
            "predictions_clv.csv":     ARTIFACTS_DIR / "predictions_clv.csv",
            "predictions_rec.csv":     ARTIFACTS_DIR / "predictions_rec.csv",
            "customer_embeddings.npy": ARTIFACTS_DIR / "customer_embeddings.npy",
        }.items():
            ok = ruta.exists()
            color = COLORS["success"] if ok else COLORS["secondary"]
            st.markdown(
                f"<p style='color:{color};font-size:0.72rem;margin:2px 0'>"
                f"{'✓' if ok else '✗'} {nombre}</p>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border-color:#2A2D3E;margin:16px 0 8px'>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:{mc};font-size:0.7rem;text-align:center'>"
            f"Olist · 93K clientes · 32K productos<br>PyTorch MultiTaskNet</p>",
            unsafe_allow_html=True,
        )

    # ── CONTENIDO PRINCIPAL ───────────────────────────────────────────────── #
    if seccion == "👤 Ficha del Cliente 360°":
        seccion_ficha_cliente()
    elif seccion == "🎯 Segmentación por Perfil":
        seccion_segmentacion()
    elif seccion == "🛍️ Recomendación y Catálogo":
        seccion_recomendacion()


if __name__ == "__main__":
    main()