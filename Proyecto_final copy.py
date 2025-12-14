# Librerías
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# CONFIGURACIÓN GENERAL (página completa, wide, ícono)
st.set_page_config(
    page_title="Superstore Giant | Rentabilidad & Descuentos",
    page_icon="🧾",
    layout="wide"
)

# PALETA / ESTILO CORPORATIVO
CORP = {
    "primary": "#1E6CCB",   # azul corporativo
    "accent":  "#0F4C81",   # azul oscuro
    "muted":   "#A7B1BC",   # gris
    "light":   "#E9EEF5",   # gris claro
    "danger":  "#C62828",   # rojo pérdidas
    "ok":      "#2E7D32",   # verde ganancias
}

PLOTLY_TEMPLATE = "simple_white"

def clean_axes(fig):
    """Quita gridlines fuertes y deja un look 'STWD' más limpio."""
    fig.update_layout(template=PLOTLY_TEMPLATE)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=CORP["light"], zeroline=False)
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(title=None)
    )
    return fig

# CARGA DE DATOS (cache) + estandarización de tipos

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    # Nota: Sample Superstore suele venir con encoding latin1
    df = pd.read_csv(path, encoding="latin1")

    # Parseo de fechas (dataset típico: Order Date, Ship Date)
    for col in ["Order Date", "Ship Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Asegurar numéricos
    for col in ["Sales", "Profit", "Discount", "Quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Campos derivados para análisis (STWD: preparar “datos para contar historia”)
    if "Order Date" in df.columns:
        df["Order_Year"] = df["Order Date"].dt.year
        df["Order_Month"] = df["Order Date"].dt.to_period("M").astype(str)

    # Margen (evitar división por cero)
    df["Profit_Margin"] = np.where(df["Sales"] > 0, df["Profit"] / df["Sales"], np.nan)

    # Bins de descuento para ver “umbral” donde se rompe la rentabilidad
    # (ajustable: aquí separamos 0%, 1–10, 11–20, 21–30, 31–40, >40)
    bins = [-0.0001, 0, 0.10, 0.20, 0.30, 0.40, 1.00]
    labels = ["0%", "1–10%", "11–20%", "21–30%", "31–40%", ">40%"]
    df["Discount_Bin"] = pd.cut(df["Discount"], bins=bins, labels=labels)

    # Bandera de pérdida
    df["Is_Loss"] = df["Profit"] < 0

    return df

# TÍTULO / CONTEXTO

st.title("Superstore Giant — Dónde los descuentos están costando rentabilidad")
st.markdown(
    "Este dashboard permite identificar **regiones, categorías y productos** donde la estrategia de descuentos "
    "está generando **ventas sin utilidad** (o pérdidas). Ajusta los filtros para replicar el diagnóstico por zona."
)

# CARGA (archivo local)

DATA_PATH = Path(__file__).parent / "superstore.csv"
df = load_data(DATA_PATH)

# SIDEBAR (filtros globales)

st.sidebar.header("Filtros globales")

# Filtro por fecha (si existe)
if "Order Date" in df.columns and df["Order Date"].notna().any():
    min_d = df["Order Date"].min().date()
    max_d = df["Order Date"].max().date()

    date_range = st.sidebar.date_input(
        "Rango de fechas (Order Date)",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d
    )
    # normalizar a tupla
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0, d1 = date_range
    else:
        d0, d1 = min_d, max_d
else:
    d0, d1 = None, None

# Multiselects típicos
def multiselect_default(label, series):
    opts = sorted(series.dropna().unique().tolist())
    return st.sidebar.multiselect(label, options=opts, default=opts)

regions_sel = multiselect_default("Región", df["Region"]) if "Region" in df.columns else []
cats_sel    = multiselect_default("Categoría", df["Category"]) if "Category" in df.columns else []
segs_sel    = multiselect_default("Segmento", df["Segment"]) if "Segment" in df.columns else []


# APLICAR FILTROS
df_f = df.copy()

if d0 and d1 and "Order Date" in df_f.columns:
    df_f = df_f[(df_f["Order Date"].dt.date >= d0) & (df_f["Order Date"].dt.date <= d1)]

if regions_sel and "Region" in df_f.columns:
    df_f = df_f[df_f["Region"].isin(regions_sel)]

if cats_sel and "Category" in df_f.columns:
    df_f = df_f[df_f["Category"].isin(cats_sel)]

if segs_sel and "Segment" in df_f.columns:
    df_f = df_f[df_f["Segment"].isin(segs_sel)]


# Guard rail: si te quedas sin datos
if df_f.empty:
    st.warning("Con esos filtros no hay datos. Ajusta región/categoría/fecha/descuento.")
    st.stop()

# KPIs (arriba)

k1, k2, k3, k4 = st.columns(4)

total_sales  = df_f["Sales"].sum()
total_profit = df_f["Profit"].sum()
margin       = (total_profit / total_sales) if total_sales > 0 else 0
avg_disc     = df_f["Discount"].mean()

with k1:
    st.metric("Total Ventas", f"${total_sales:,.0f}")
with k2:
    st.metric("Total Utilidad", f"${total_profit:,.0f}")
with k3:
    st.metric("Margen (%)", f"{margin*100:,.1f}%")
with k4:
    st.metric("Descuento promedio", f"{avg_disc*100:,.1f}%")
    
# GRÁFICOS PRINCIPALES
# PESTAÑAS DE ANÁLISIS

tab1, tab2, tab3 = st.tabs([
    "📉 Descuento vs Utilidad",
    "💰 Ventas altas, pérdidas",
    "🏷️ Estrategia de precios"
])

# PESTAÑA 1 — Relación directa entre descuento y utilidad
with tab1:
    st.subheader("Relación directa entre nivel de descuento y utilidad")

    c1, c2 = st.columns(2)

    # Gráfico 1: Discount vs Profit (scatter)
    fig1 = px.scatter(
        df_f,
        x="Discount",
        y="Profit",
        color="Category",
        size="Sales",
        hover_data=["Sub-Category"],
        title="Descuento aplicado vs Utilidad generada",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig1.update_traces(marker=dict(opacity=0.6))
    fig1 = clean_axes(fig1)
    fig1.update_yaxes(showgrid=False)
    fig1.update_xaxes(showgrid=False)
    fig1.add_hline(
    y=0,
    line_width=2,
    line_color=CORP["muted"],  # o un gris/blanco suave
    opacity=1,
    layer="above"
    )
    
    c1.plotly_chart(fig1, use_container_width=True)

    c1.info(
        "**Insight:** A medida que el descuento aumenta, la utilidad tiende a disminuir. "
        "Existen múltiples transacciones con descuentos altos que generan pérdidas, "
        "lo que indica un umbral de descuento donde la rentabilidad se rompe."
    )

    # Gráfico 2: Profit promedio por Discount_Bin
    disc_bin_profit = (
        df_f.groupby("Discount_Bin", observed=True)
        .agg(Avg_Profit=("Profit", "mean"))
        .reset_index()
    )

    fig2 = px.bar(
        disc_bin_profit,
        x="Discount_Bin",
        y="Avg_Profit",
        title="Utilidad promedio por rango de descuento",
        text_auto=".2s",
        color="Avg_Profit",
        color_continuous_scale=["#C62828", "#E9EEF5", "#2E7D32"]
    )
    fig2 = clean_axes(fig2)
    fig2.update_yaxes(
    title=None,        # elimina título eje Y
    showticklabels=False,  # elimina números del eje Y
    showgrid=False     # elimina líneas horizontales
    )

    fig2.update_layout(
    yaxis=dict(zeroline=False)  # elimina línea cero
)

    c2.plotly_chart(fig2, use_container_width=True)

    c2.info(
        "**Insight:** Los rangos de descuento superiores al 20–30% presentan utilidades promedio negativas, "
        "confirmando que los descuentos agresivos no están siendo compensados por el volumen de ventas."
    )

# PESTAÑA 2 — Categorías / productos con ventas altas y pérdidas

with tab2:
    st.subheader("Productos con alto volumen de ventas que generan pérdidas")

    c1, c2 = st.columns(2)

    # Gráfico 3: Heatmap Sub-Category vs Discount_Bin (Avg Profit)

    heatmap_data = (
        df_f.groupby(["Sub-Category", "Discount_Bin"], observed=True)
        .agg(Avg_Profit=("Profit", "mean"))
        .reset_index()
    )

    fig3 = px.density_heatmap(
        heatmap_data,
        x="Discount_Bin",
        y="Sub-Category",
        z="Avg_Profit",
        color_continuous_scale=[
            CORP["danger"],   # pérdidas
            CORP["light"],    # neutral
            CORP["ok"]        # ganancias
        ],
        title="Utilidad promedio por subcategoría y nivel de descuento"
    )

    fig3.update_layout(
        coloraxis_colorbar=dict(title="Utilidad promedio")
    )

    fig3 = clean_axes(fig3)

    c1.plotly_chart(fig3, use_container_width=True)

    c1.info(
        "**Insight:** El heatmap muestra que varias subcategorías son rentables únicamente con descuentos bajos. "
        "A partir de ciertos rangos de descuento, la utilidad promedio se vuelve negativa, lo que indica que "
        "la política de descuentos está comprometiendo la rentabilidad de forma estructural."
    )

    # Gráfico 4: Top 10 subcategorías con mayor pérdida
    subcat_perf = (
        df_f.groupby(["Category", "Sub-Category"])
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )
    loss_subcats = (
        subcat_perf.sort_values("Profit")
        .head(10)
    )

    fig4 = px.bar(
        loss_subcats,
        x="Profit",
        y="Sub-Category",
        orientation="h",
        title="Top 10 subcategorías con mayores pérdidas",
        color="Profit",
        color_continuous_scale=["#C62828", "#E9EEF5"]
    )
    fig4 = clean_axes(fig4)
    fig4.update_layout(
    coloraxis_showscale=False
    )
    
    fig4.update_xaxes(title=None)
    fig4.update_yaxes(title=None)
    
    fig4.update_xaxes(showgrid=False)
    fig4.update_yaxes(showgrid=False)
    
    fig4.update_layout(
    xaxis=dict(zeroline=False)
    )

    c2.plotly_chart(fig4, use_container_width=True)

    c2.info(
        "**Insight:** Estas subcategorías concentran las mayores pérdidas operativas y deben ser priorizadas "
        "para revisión de precios, descuentos o incluso racionalización del portafolio."
    )
    
# PESTAÑA 3 — Evaluación de la estrategia de precios
with tab3:
    st.subheader("Evaluación de la estrategia de precios por categoría")


    # Gráfico 6: Profit Margin vs Discount por Category
    fig6 = px.scatter(
        df_f,
        x="Discount",
        y="Profit_Margin",
        color="Category",
        title="Margen de utilidad vs Descuento",
        hover_data=["Sub-Category"],
        color_discrete_sequence=px.colors.qualitative.Dark2
    )
    fig6.add_hline(y=0, line_dash="dash", line_color=CORP["muted"])
    fig6 = clean_axes(fig6)
    fig6.update_xaxes(showgrid=False)
    fig6.update_yaxes(showgrid=False)
    
    fig6.add_hline(
    y=0,
    line_dash="dash",
    line_color=CORP["muted"]
    )

    fig6.update_traces(
    marker=dict(size=10, opacity=0.75)
    )

    fig6.update_traces(
    marker=dict(size=11, opacity=0.8)
    )

    st.plotly_chart(fig6, use_container_width=True)
    
    st.info(
        "*Insight:* En varias categorías, el aumento del descuento reduce directamente el margen de utilidad. "
        "Solo ciertas líneas toleran descuentos sin comprometer la rentabilidad."
    )
    
    # Gráfico 7: Serie temporal de ventas y utilidad
    st.subheader("Crecimiento en ventas no siempre significa crecimiento en utilidad")
    if "Order_Month" in df_f.columns:
        ts = (df_f.groupby("Order_Month", as_index=False)
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")))
        # Orden cronológico (Order_Month es YYYY-MM)
        ts = ts.sort_values("Order_Month")

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=ts["Order_Month"], y=ts["Sales"],
            mode="lines+markers", name="Ventas",
            line=dict(color=CORP["primary"])
        ))
        fig_ts.add_trace(go.Scatter(
            x=ts["Order_Month"], y=ts["Profit"],
            mode="lines+markers", name="Utilidad",
            line=dict(color=CORP["muted"])
        ))
        fig_ts.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title=None, yaxis_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        fig_ts.update_yaxes(showgrid=False, zeroline=False)
        fig_ts.update_xaxes(showgrid=False)

        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No hay columna de fecha mensual (Order Date/Order_Month) para la serie temporal.")
        
    st.info(
    "Insight: Las ventas muestran una tendencia al alza, pero la utilidad no crece al mismo ritmo y es más volátil. "
    "Esto sugiere que el crecimiento se está comprando con descuentos/costos: vender más no está garantizando rentabilidad."
    )

# Mapa: Utilidad por Estado (dónde duele)
st.subheader("Geografía de la rentabilidad: estados con pérdidas recurrentes")
if "State" in df_f.columns:
    # Sample Superstore usa nombres de estado; Plotly choropleth usa abreviaciones.
    # Creamos un mapping mínimo con state_code si está disponible; si no, usamos una tabla.
    # (Esto es “base”: si tu profe exige exactitud total, completamos el mapping de 50 estados.)
    state_map = {
        "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO",
        "Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID",
        "Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
        "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS",
        "Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ",
        "New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK",
        "Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
        "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA",
        "West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY","District of Columbia":"DC"
    }
    geo = (df_f.groupby("State", as_index=False)
           .agg(Sales=("Sales","sum"), Profit=("Profit","sum")))
    geo["state_code"] = geo["State"].map(state_map)

    geo_ok = geo.dropna(subset=["state_code"])
    if not geo_ok.empty:
        fig_map = px.choropleth(
            geo_ok,
            locations="state_code",
            locationmode="USA-states",
            color="Profit",
            scope="usa",
            hover_name="State",
            hover_data={"Sales":":.0f","Profit":":.0f"},
            color_continuous_scale=[CORP["danger"], CORP["light"], CORP["primary"]]
        )
        fig_map.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No se pudo mapear State -> abreviación. (Se completa el mapping si tu dataset tiene estados distintos).")
else:
    st.info("No existe columna State para construir el mapa.")

st.divider()

# ============================================================
# 9) INSIGHT FINAL (guiar decisión)
# ============================================================

# 1) Peores subcategorías por Profit (pérdida)
if "Sub-Category" in df_f.columns:
    loss_sub = (df_f.groupby("Sub-Category", as_index=False)
                .agg(Sales=("Sales","sum"), Profit=("Profit","sum"), AvgDisc=("Discount","mean")))
    loss_sub = loss_sub.sort_values("Profit").head(3)

    worst = ", ".join([f"{r['Sub-Category']} (${r['Profit']:,.0f})" for _, r in loss_sub.iterrows()])
else:
    worst = "N/A (no hay Sub-Category)"

# 2) Bin de descuento con peor mediana de Profit (para hablar de “umbral”)
bin_med = (df_f.dropna(subset=["Discount_Bin"])
           .groupby("Discount_Bin", as_index=False)["Profit"].median()
           .sort_values("Profit"))
worst_bin = bin_med.iloc[0]["Discount_Bin"] if not bin_med.empty else "N/A"

st.success(
    f"Conclusión: con los filtros actuales, el tramo de descuento más riesgoso es **{worst_bin}** "
    f"(mediana de utilidad más baja). Además, las sub-categorías con peor contribución son: **{worst}**. "
    "Recomendación: **limitar descuentos en ese umbral**, y rediseñar pricing/promos para las sub-categorías en pérdida "
    "antes de seguir empujando volumen."
)