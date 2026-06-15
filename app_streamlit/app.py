import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Adidas US Sales Dashboard",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f0f0f; color: #f0f0f0; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1a1a; border-right: 1px solid #333; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e1e1e, #2a2a2a);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    .kpi-label { font-size: 13px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #fff; }
    .kpi-delta-pos { font-size: 13px; color: #00e676; margin-top: 4px; }
    .kpi-delta-neg { font-size: 13px; color: #ff5252; margin-top: 4px; }
    .kpi-delta-neu { font-size: 13px; color: #aaa; margin-top: 4px; }

    /* Section headers */
    .section-header {
        font-size: 16px; font-weight: 600; color: #fff;
        border-left: 4px solid #e63946;
        padding-left: 10px; margin: 24px 0 12px 0;
    }

    /* Prediction card */
    .pred-box {
        background: linear-gradient(135deg, #1a2a1a, #1e3a1e);
        border: 1px solid #2e7d32;
        border-radius: 12px;
        padding: 22px; text-align: center;
    }
    .pred-value { font-size: 36px; font-weight: 800; color: #69f0ae; }

    /* Adidas header bar */
    .header-bar {
        background: linear-gradient(90deg, #e63946, #c1121f);
        padding: 14px 28px; border-radius: 10px;
        margin-bottom: 20px;
        display: flex; align-items: center; gap: 14px;
    }
    .header-title { font-size: 26px; font-weight: 800; color: #fff; margin: 0; }
    .header-sub { font-size: 13px; color: #ffd; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)

    # Try relative paths first, then fallback for demo
    paths_df = [
        os.path.join(base, "../data/processed/Adidas_US_Sales_procesed.csv"),
        os.path.join(base, "Adidas_US_Sales_procesed.csv"),
    ]
    paths_X = [
        os.path.join(base, "../data/processed/X_preprocessed.csv"),
        os.path.join(base, "X_preprocessed.csv"),
    ]
    paths_y = [
        os.path.join(base, "../data/processed/y_target.csv"),
        os.path.join(base, "y_target.csv"),
    ]

    df = X = y = None
    for p in paths_df:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    for p in paths_X:
        if os.path.exists(p):
            X = pd.read_csv(p)
            break
    for p in paths_y:
        if os.path.exists(p):
            y = pd.read_csv(p)["Total Sales"]
            break

    if df is None or X is None or y is None:
        st.error("⚠️ No se encontraron los archivos de datos. Ajusta las rutas en `load_data()`.")
        st.stop()

    df["Invoice Date"] = pd.to_datetime(df["Invoice Date"])
    df["Month"] = df["Invoice Date"].dt.month
    df["Year"] = df["Invoice Date"].dt.year
    df["DayOfWeek"] = df["Invoice Date"].dt.dayofweek
    df["Is_Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["Month_Name"] = df["Invoice Date"].dt.strftime("%b %Y")
    return df, X, y


# ─────────────────────────────────────────────
# RUTA DE MODELOS — ajusta aquí si cambia la ubicación
# ─────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")

# Mapeo nombre legible → archivo .pkl
MODEL_FILES = {
    "📐 Linear Regression":  "Linear_Regression.pkl",
    "🔵 KNN ":               "KNN.pkl",
    "🌲 Random Forest" :     "Random_Forest.pkl",
    "⚡ SVR":                "SVR.pkl",
    "🚀 XGBoost":            "XGBoost.pkl",
}

def get_available_models(model_dir: str) -> dict:
    """
    Escanea model_dir y devuelve sólo los modelos cuyo .pkl existe.
    Si no existe ninguno, devuelve dict vacío y avisa en pantalla.
    """
    available = {}
    missing   = []
    for label, fname in MODEL_FILES.items():
        full_path = os.path.join(model_dir, fname)
        if os.path.exists(full_path):
            available[label] = full_path
        else:
            missing.append(fname)
    return available, missing


@st.cache_resource
def load_single_model(path: str):
    """Carga un único .pkl con caché. Se llama sólo cuando el usuario elige modelo."""
    with open(path, "rb") as f:
        return pickle.load(f)


df, X, y = load_data()
available_models, missing_models = get_available_models(MODEL_DIR)

# ─────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Filtros")
    st.markdown("---")

    # Date range
    min_date = df["Invoice Date"].min().date()
    max_date = df["Invoice Date"].max().date()
    date_range = st.date_input(
        "📅 Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Retailer
    retailers = ["Todos"] + sorted(df["Retailer"].unique().tolist())
    sel_retailer = st.multiselect("🏪 Retailer", retailers, default=["Todos"])

    # Region
    regions = ["Todas"] + sorted(df["Region"].unique().tolist())
    sel_region = st.multiselect("🗺️ Región", regions, default=["Todas"])

    # Product
    products = ["Todos"] + sorted(df["Product"].unique().tolist())
    sel_product = st.multiselect("👟 Producto", products, default=["Todos"])

    # Sales Method
    methods = ["Todos"] + sorted(df["Sales Method"].unique().tolist())
    sel_method = st.multiselect("💳 Método de Venta", methods, default=["Todos"])

    # Temporal
    st.markdown("### ⏱️ Temporal")
    years_avail = sorted(df["Year"].unique().tolist())
    sel_years = st.multiselect("Año", years_avail, default=years_avail)
    weekend_opt = st.radio("Fin de semana", ["Todos", "Solo fin de semana", "Solo semana"], index=0)

    st.markdown("---")
    st.caption("👟 Adidas US Sales · ML Dashboard")

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
mask = (
    (df["Invoice Date"].dt.date >= start_date) &
    (df["Invoice Date"].dt.date <= end_date) &
    (df["Year"].isin(sel_years))
)
if "Todos" not in sel_retailer:
    mask &= df["Retailer"].isin(sel_retailer)
if "Todas" not in sel_region:
    mask &= df["Region"].isin(sel_region)
if "Todos" not in sel_product:
    mask &= df["Product"].isin(sel_product)
if "Todos" not in sel_method:
    mask &= df["Sales Method"].isin(sel_method)
if weekend_opt == "Solo fin de semana":
    mask &= df["Is_Weekend"] == 1
elif weekend_opt == "Solo semana":
    mask &= df["Is_Weekend"] == 0

dff = df[mask].copy()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <div>
    <p class="header-title">👟 Adidas US Sales Dashboard</p>
    <p class="header-sub">Panel de Ventas · Machine Learning · Total Sales como Target</p>
  </div>
</div>
""", unsafe_allow_html=True)

if dff.empty:
    st.warning("⚠️ No hay datos con los filtros seleccionados. Ajusta los filtros en la barra lateral.")
    st.stop()

# ─────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────
total_sales = dff["Total Sales"].sum()
total_units = dff["Units Sold"].sum()
avg_price = dff["Price per Unit"].mean()
total_profit = dff["Operating Profit"].sum()
avg_margin = dff["Operating Margin"].mean() * 100
num_transactions = len(dff)

# Compare vs full dataset for delta
total_sales_all = df["Total Sales"].sum()
pct_vs_total = ((total_sales - total_sales_all) / total_sales_all * 100) if total_sales_all else 0

st.markdown('<p class="section-header">📊 KPIs Principales</p>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

def kpi_card(col, label, value, delta_text, delta_pos=True):
    cls = "kpi-delta-pos" if delta_pos else "kpi-delta-neg"
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="{cls}">{delta_text}</div>
    </div>""", unsafe_allow_html=True)

with k1:
    kpi_card(k1, "💰 Total Sales", f"${total_sales/1e6:.1f}M",
             f"{pct_vs_total:+.1f}% vs total", pct_vs_total >= 0)
with k2:
    kpi_card(k2, "📦 Units Sold", f"{total_units:,}",
             f"{num_transactions:,} transacciones", True)
with k3:
    kpi_card(k3, "💵 Precio Medio", f"${avg_price:.0f}",
             "por unidad", True)
with k4:
    kpi_card(k4, "📈 Op. Profit", f"${total_profit/1e6:.1f}M",
             f"Margen: {avg_margin:.1f}%", total_profit >= 0)
with k5:
    top_retailer = dff.groupby("Retailer")["Total Sales"].sum().idxmax()
    kpi_card(k5, "🏆 Top Retailer", top_retailer[:12],
             "por ventas totales", True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 1: Time Series + Sales Method
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">📈 Ventas a lo largo del tiempo</p>', unsafe_allow_html=True)
col_ts, col_method = st.columns([3, 1])

with col_ts:
    ts = (dff.groupby(["Invoice Date", "Product"])["Total Sales"]
          .sum().reset_index())
    fig_ts = px.line(
        ts, x="Invoice Date", y="Total Sales", color="Product",
        title="Total Sales por Producto en el Tiempo",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_ts.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        legend=dict(orientation="h", y=-0.25, x=0),
        margin=dict(t=40, b=10),
        hovermode="x unified",
        yaxis_tickformat="$,.0f",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

with col_method:
    method_sales = dff.groupby("Sales Method")["Total Sales"].sum().reset_index()
    fig_method = px.pie(
        method_sales, values="Total Sales", names="Sales Method",
        title="Ventas por Método",
        template="plotly_dark",
        color_discrete_sequence=["#e63946", "#457b9d", "#2a9d8f"],
        hole=0.4,
    )
    fig_method.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        margin=dict(t=40, b=10),
        legend=dict(orientation="h", y=-0.1),
    )
    st.plotly_chart(fig_method, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 2: Region map + Product bars
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">🗺️ Ventas por Región y Producto</p>', unsafe_allow_html=True)
col_reg, col_prod = st.columns(2)

with col_reg:
    region_sales = dff.groupby("Region")["Total Sales"].sum().reset_index().sort_values("Total Sales", ascending=True)
    fig_reg = px.bar(
        region_sales, x="Total Sales", y="Region", orientation="h",
        title="Total Sales por Región",
        template="plotly_dark",
        color="Total Sales",
        color_continuous_scale="Reds",
    )
    fig_reg.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        margin=dict(t=40, b=10),
        coloraxis_showscale=False,
        xaxis_tickformat="$,.0f",
    )
    st.plotly_chart(fig_reg, use_container_width=True)

with col_prod:
    prod_sales = dff.groupby("Product")["Total Sales"].sum().reset_index().sort_values("Total Sales", ascending=True)
    fig_prod = px.bar(
        prod_sales, x="Total Sales", y="Product", orientation="h",
        title="Total Sales por Categoría de Producto",
        template="plotly_dark",
        color="Total Sales",
        color_continuous_scale="Blues",
    )
    fig_prod.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        margin=dict(t=40, b=10),
        coloraxis_showscale=False,
        xaxis_tickformat="$,.0f",
    )
    st.plotly_chart(fig_prod, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 3: Retailer ranking + Heatmap
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">🏪 Retailer & Heatmap Temporal</p>', unsafe_allow_html=True)
col_ret, col_heat = st.columns(2)

with col_ret:
    ret_sales = (dff.groupby("Retailer")
                 .agg(Total_Sales=("Total Sales", "sum"),
                      Units=("Units Sold", "sum"),
                      Transactions=("Total Sales", "count"))
                 .reset_index()
                 .sort_values("Total_Sales", ascending=False))
    ret_sales["Total_Sales_fmt"] = ret_sales["Total_Sales"].apply(lambda x: f"${x/1e6:.2f}M")
    ret_sales["Units_fmt"] = ret_sales["Units"].apply(lambda x: f"{x:,}")

    fig_ret = px.bar(
        ret_sales, x="Retailer", y="Total_Sales",
        title="Total Sales por Retailer",
        template="plotly_dark",
        color="Retailer",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        text="Total_Sales_fmt",
    )
    fig_ret.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        margin=dict(t=40, b=10),
        showlegend=False,
        yaxis_tickformat="$,.0f",
    )
    fig_ret.update_traces(textposition="outside")
    st.plotly_chart(fig_ret, use_container_width=True)

with col_heat:
    heat_data = dff.groupby(["Year", "Month"])["Total Sales"].sum().reset_index()
    heat_pivot = heat_data.pivot(index="Month", columns="Year", values="Total Sales").fillna(0)
    month_labels = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    heat_pivot.index = [month_labels[i-1] for i in heat_pivot.index]

    fig_heat = px.imshow(
        heat_pivot,
        title="Heatmap: Ventas por Mes y Año",
        template="plotly_dark",
        color_continuous_scale="Reds",
        aspect="auto",
        text_auto=".2s",
    )
    fig_heat.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 4: Operating Profit scatter + Weekend vs Weekday
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">📊 Análisis Adicional</p>', unsafe_allow_html=True)
col_sc, col_wk = st.columns(2)

with col_sc:
    fig_sc = px.scatter(
        dff.sample(min(2000, len(dff))), x="Units Sold", y="Total Sales",
        color="Product", size="Operating Profit",
        title="Units Sold vs Total Sales (muestra)",
        template="plotly_dark",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_sc.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        margin=dict(t=40, b=10),
        legend=dict(orientation="h", y=-0.3),
        yaxis_tickformat="$,.0f",
    )
    st.plotly_chart(fig_sc, use_container_width=True)

with col_wk:
    dff["Tipo de Día"] = dff["Is_Weekend"].map({1: "Fin de semana", 0: "Semana"})
    wk = dff.groupby(["Tipo de Día", "Sales Method"])["Total Sales"].sum().reset_index()
    fig_wk = px.bar(
        wk, x="Tipo de Día", y="Total Sales", color="Sales Method",
        title="Ventas: Semana vs Fin de Semana por Método",
        template="plotly_dark",
        barmode="group",
        color_discrete_sequence=["#e63946", "#457b9d", "#2a9d8f"],
    )
    fig_wk.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        margin=dict(t=40, b=10),
        yaxis_tickformat="$,.0f",
    )
    st.plotly_chart(fig_wk, use_container_width=True)

# ─────────────────────────────────────────────
# TOP TABLES
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">📋 Rankings y Tablas</p>', unsafe_allow_html=True)
col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
    st.markdown("**🏪 Top Retailers**")
    top_ret = (dff.groupby("Retailer")["Total Sales"].sum()
               .reset_index().sort_values("Total Sales", ascending=False))
    top_ret["Total Sales"] = top_ret["Total Sales"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(top_ret, hide_index=True, use_container_width=True)

with col_t2:
    st.markdown("**👟 Top Productos**")
    top_prod = (dff.groupby("Product")["Total Sales"].sum()
                .reset_index().sort_values("Total Sales", ascending=False))
    top_prod["Total Sales"] = top_prod["Total Sales"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(top_prod, hide_index=True, use_container_width=True)

with col_t3:
    st.markdown("**🗺️ Top Regiones**")
    top_reg = (dff.groupby("Region")["Total Sales"].sum()
               .reset_index().sort_values("Total Sales", ascending=False))
    top_reg["Total Sales"] = top_reg["Total Sales"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(top_reg, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────
# ML PREDICTION SECTION
# ─────────────────────────────────────────────
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

st.markdown("---")
st.markdown('<p class="section-header">🤖 Predicción de Total Sales (ML)</p>', unsafe_allow_html=True)

# ── Metadatos de cada modelo ─────────────────────────────────────────────────
MODEL_META = {
    "Linear Regression": {
        "file":  "Linear_Regression.pkl",
        "emoji": "📐",
        "color": "#457b9d",
        "desc":  "Modelo lineal simple. Muy rápido e interpretable. Ideal como baseline.",
        "tags":  ["Lineal", "Interpretable", "Baseline"],
    },
    "KNN": {
        "file":  "KNN.pkl",
        "emoji": "🔵",
        "color": "#2a9d8f",
        "desc":  "K-Nearest Neighbors con hiperparámetros optimizados. Predice por proximidad.",
        "tags":  ["No paramétrico", "Vecinos", "Hyper-tuned"],
    },
    "Random Forest": {
        "file":  "Random_Forest.pkl",
        "emoji": "🌲",
        "color": "#2d6a4f",
        "desc":  "Ensemble de árboles de decisión. Robusto, preciso y con feature importance.",
        "tags":  ["Ensemble", "Alta precisión", "Hyper-tuned"],
    },
    "SVR": {
        "file":  "SVR.pkl",
        "emoji": "⚡",
        "color": "#e9c46a",
        "desc":  "Support Vector Regression optimizado. Eficaz en espacios de alta dimensión.",
        "tags":  ["SVM", "Kernel", "Hyper-tuned"],
    },
    "XGBoost": {
        "file":  "XGBoost.pkl",
        "emoji": "🚀",
        "color": "#e63946",
        "desc":  "Gradient Boosting extremo. Generalmente el mejor rendimiento en datos tabulares.",
        "tags":  ["Boosting", "Top rendimiento", "Hyper-tuned"],
    },
}


def is_model_fitted(mdl) -> bool:
    """Devuelve True si el modelo tiene atributos de entrenamiento."""
    try:
        check_is_fitted(mdl)
        return True
    except NotFittedError:
        return False
    except Exception:
        # XGBoost no lanza NotFittedError estándar; comprobamos n_features_in_
        return hasattr(mdl, "n_features_in_") or hasattr(mdl, "feature_importances_")


def load_and_ensure_fitted(path: str, X_train, y_train):
    """
    Carga el .pkl. Si el modelo no está fitted, lo reentrena con X, y
    y lo devuelve junto con un flag indicando si fue necesario reentrenar.
    """
    with open(path, "rb") as f:
        mdl = pickle.load(f)

    retrained = False
    if not is_model_fitted(mdl):
        mdl.fit(X_train, y_train)
        retrained = True
    return mdl, retrained


# ── Escanear carpeta de modelos ──────────────────────────────────────────────
found_models = {}   # label → path
missing_files = []

for label, meta in MODEL_META.items():
    path = os.path.join(MODEL_DIR, meta["file"])
    if os.path.exists(path):
        found_models[label] = path
    else:
        missing_files.append(meta["file"])

if missing_files:
    st.warning(
        f"⚠️ No se encontraron los siguientes archivos en `{MODEL_DIR}`:\n"
        + "  ".join(f"`{f}`" for f in missing_files)
    )

if not found_models:
    st.error(
        f"❌ No hay ningún modelo `.pkl` en:\n\n`{MODEL_DIR}`\n\n"
        "Asegúrate de que la ruta es correcta y los modelos están entrenados."
    )
    st.stop()

# ── CSS para las tarjetas de modelo ─────────────────────────────────────────
st.markdown("""
<style>
.model-card {
    border: 2px solid #333; border-radius: 14px;
    padding: 16px 14px; text-align: center;
    cursor: pointer; transition: all .2s;
    background: #1e1e1e; margin-bottom: 4px;
}
.model-card:hover { border-color: #e63946; background: #2a2a2a; }
.model-card.selected { border-color: #e63946 !important; background: #2a1a1a !important; }
.model-tag {
    display: inline-block; font-size: 10px; padding: 2px 8px;
    border-radius: 20px; background: #333; color: #ccc;
    margin: 2px 1px;
}
.metric-box {
    background: #1e1e1e; border: 1px solid #333; border-radius: 10px;
    padding: 14px; text-align: center;
}
.metric-box .val  { font-size: 24px; font-weight: 700; color: #fff; }
.metric-box .lbl  { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
.metric-box .sub  { font-size: 11px; color: #aaa; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── 1. Menú visual de selección de modelo ───────────────────────────────────
st.markdown("#### 1️⃣ Selecciona el modelo")

n_cols = len(found_models)
model_cols = st.columns(n_cols)
model_labels = list(found_models.keys())

# Guardamos selección en session_state
if "sel_model" not in st.session_state or st.session_state.sel_model not in found_models:
    st.session_state.sel_model = model_labels[0]

for i, label in enumerate(model_labels):
    meta = MODEL_META[label]
    with model_cols[i]:
        selected = st.session_state.sel_model == label
        border_style = f"border-color:{meta['color']}; background:#1a1a1a;" if selected else ""
        tags_html = "".join(f'<span class="model-tag">{t}</span>' for t in meta["tags"])
        st.markdown(f"""
        <div class="model-card" style="{border_style}">
          <div style="font-size:28px">{meta['emoji']}</div>
          <div style="font-weight:700;font-size:13px;color:#fff;margin:4px 0">{label}</div>
          <div style="font-size:11px;color:#aaa;min-height:38px">{meta['desc'][:60]}…</div>
          <div style="margin-top:6px">{tags_html}</div>
        </div>""", unsafe_allow_html=True)
        if st.button(
            f"{'✅ Seleccionado' if selected else 'Seleccionar'}",
            key=f"btn_{label}",
            use_container_width=True,
            type="primary" if selected else "secondary",
        ):
            st.session_state.sel_model = label
            st.rerun()

# ── 2. Cargar modelo seleccionado y mostrar métricas ────────────────────────
sel_label = st.session_state.sel_model
sel_meta  = MODEL_META[sel_label]
sel_path  = found_models[sel_label]

with st.spinner(f"Cargando {sel_label}..."):
    model, was_retrained = load_and_ensure_fitted(sel_path, X, y)

st.markdown(f"---\n#### 2️⃣ Métricas — {sel_meta['emoji']} {sel_label}")

if was_retrained:
    st.info(
        "ℹ️ El archivo `.pkl` contenía el modelo **sin entrenar** (solo la configuración). "
        f"Se ha reentrenado automáticamente con el dataset completo para calcular las métricas. "
        f"Si quieres persistir el modelo entrenado, guárdalo en `{sel_path}`."
    )

# Calcular métricas
try:
    y_pred_full = model.predict(X)
    r2_val   = r2_score(y, y_pred_full)
    mae_val  = mean_absolute_error(y, y_pred_full)
    rmse_val = np.sqrt(mean_squared_error(y, y_pred_full))
    mape_val = np.mean(np.abs((y - y_pred_full) / y.replace(0, np.nan))) * 100

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(f"""<div class="metric-box">
        <div class="lbl">R² Score</div>
        <div class="val" style="color:{'#69f0ae' if r2_val>=0.8 else '#ffb74d' if r2_val>=0.6 else '#ff5252'}">{r2_val:.4f}</div>
        <div class="sub">{'Excelente ✅' if r2_val>=0.9 else 'Bueno 👍' if r2_val>=0.7 else 'Mejorable ⚠️'}</div>
    </div>""", unsafe_allow_html=True)
    mc2.markdown(f"""<div class="metric-box">
        <div class="lbl">MAE</div>
        <div class="val">${mae_val:,.0f}</div>
        <div class="sub">Error absoluto medio</div>
    </div>""", unsafe_allow_html=True)
    mc3.markdown(f"""<div class="metric-box">
        <div class="lbl">RMSE</div>
        <div class="val">${rmse_val:,.0f}</div>
        <div class="sub">Raíz error cuadrático</div>
    </div>""", unsafe_allow_html=True)
    mc4.markdown(f"""<div class="metric-box">
        <div class="lbl">MAPE</div>
        <div class="val">{mape_val:.1f}%</div>
        <div class="sub">Error porcentual medio</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfico Real vs Predicho (muestra)
    col_chart, col_feat = st.columns(2)
    with col_chart:
        sample_idx = np.random.choice(len(y), min(500, len(y)), replace=False)
        fig_rv = go.Figure()
        fig_rv.add_trace(go.Scatter(
            x=y.iloc[sample_idx], y=y_pred_full[sample_idx],
            mode="markers", marker=dict(color=sel_meta["color"], opacity=0.6, size=5),
            name="Predicciones"
        ))
        lims = [min(y.min(), y_pred_full.min()), max(y.max(), y_pred_full.max())]
        fig_rv.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                    line=dict(color="#fff", dash="dash", width=1), name="Ideal"))
        fig_rv.update_layout(
            title="Real vs Predicho (muestra 500)",
            template="plotly_dark", plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
            xaxis_title="Real ($)", yaxis_title="Predicho ($)",
            margin=dict(t=40, b=10), xaxis_tickformat="$,.0f", yaxis_tickformat="$,.0f",
        )
        st.plotly_chart(fig_rv, use_container_width=True)

    with col_feat:
        if hasattr(model, "feature_importances_"):
            fi = (pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
                  .sort_values("Importance", ascending=False).head(12))
            fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                            title="Importancia de Features (Top 12)",
                            template="plotly_dark",
                            color="Importance", color_continuous_scale="Reds")
            fig_fi.update_layout(plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
                                  coloraxis_showscale=False, margin=dict(t=40, b=10))
            st.plotly_chart(fig_fi, use_container_width=True)
        elif hasattr(model, "coef_"):
            coef_vals = model.coef_ if model.coef_.ndim == 1 else model.coef_[0]
            coefs = (pd.DataFrame({"Feature": X.columns, "Coef (abs)": np.abs(coef_vals)})
                     .sort_values("Coef (abs)", ascending=False).head(12))
            fig_coef = px.bar(coefs, x="Coef (abs)", y="Feature", orientation="h",
                              title="Coeficientes del Modelo (Top 12)",
                              template="plotly_dark",
                              color="Coef (abs)", color_continuous_scale="Blues")
            fig_coef.update_layout(plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
                                    coloraxis_showscale=False, margin=dict(t=40, b=10))
            st.plotly_chart(fig_coef, use_container_width=True)
        else:
            st.info("ℹ️ Este modelo no expone importancia de features ni coeficientes.")

except Exception as e:
    st.error(f"❌ Error al evaluar el modelo: {e}")

# ── 3. Formulario de predicción ──────────────────────────────────────────────
st.markdown("---\n#### 3️⃣ Haz una predicción")
col_m1, col_m2 = st.columns([1, 2])

with col_m1:
    pred_retailer = st.selectbox(
        "🏪 Retailer",
        ["Foot Locker", "West Gear", "Sports Direct", "Kohl's", "Amazon", "Walmart"]
    )
    pred_region = st.selectbox(
        "🗺️ Región",
        ["West", "Northeast", "Midwest", "South", "Southeast"]
    )
    pred_product = st.selectbox(
        "👟 Producto",
        ["Men's Street Footwear", "Men's Athletic Footwear", "Men's Apparel",
         "Women's Street Footwear", "Women's Apparel", "Women's Athletic Footwear"]
    )
    pred_method = st.selectbox(
        "💳 Método de Venta",
        ["Online", "Outlet", "In-store"]
    )
    pred_price = st.slider("💵 Precio por Unidad ($)", 20, 200, 50, step=5)
    pred_month = st.selectbox(
        "📅 Mes",
        options=list(range(1, 13)),
        format_func=lambda m: ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                               "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"][m-1]
    )
    pred_year = st.selectbox("📆 Año", [2020, 2021, 2022, 2023])
    pred_dow = st.select_slider(
        "📅 Día de la Semana",
        options=[0,1,2,3,4,5,6],
        format_func=lambda d: ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"][d],
        value=2,
    )
    pred_is_weekend = 1 if pred_dow >= 5 else 0
    st.caption(f"🗓️ Is_Weekend: {'✅ Sí (Sáb/Dom)' if pred_is_weekend else '❌ No (Lun–Vie)'}")

with col_m2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Predecir Total Sales", type="primary", use_container_width=True):
        row = {c: 0 for c in X.columns.tolist()}
        row.update({
            "Price per Unit": pred_price,
            "Month":          pred_month,
            "Year":           pred_year,
            "DayOfWeek":      pred_dow,
            "Is_Weekend":     pred_is_weekend,
        })
        for key in [f"Retailer_{pred_retailer}", f"Region_{pred_region}",
                    f"Product_{pred_product}",   f"Sales Method_{pred_method}"]:
            if key in row:
                row[key] = 1

        X_input = pd.DataFrame([row])
        try:
            prediction = model.predict(X_input)[0]
            st.markdown(f"""
            <div class="pred-box">
              <div class="kpi-label">💰 Total Sales Predicho — {sel_meta['emoji']} {sel_label}</div>
              <div class="pred-value">${prediction:,.0f}</div>
              <div class="kpi-delta-pos">R² del modelo: {r2_val:.4f} · MAE: ${mae_val:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error al predecir: {e}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>👟 Adidas US Sales Dashboard · Construido con Streamlit & Plotly · "
    f"Dataset: {len(df):,} registros · Filtrado: {len(dff):,} registros</small></center>",
    unsafe_allow_html=True
)
