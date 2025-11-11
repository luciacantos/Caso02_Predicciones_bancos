import os
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import metrics_segment, plot_series

st.set_page_config(page_title="Predicción BBVA & Santander", page_icon="💼", layout="wide")

# ===== Estilo extra (negro + dorado) =====
st.markdown("""
<style>
:root { --gold:#CBA135; --text:#EDEDED; --muted:#A9B0B8; }
html, body, [class^="css"] {
  background: radial-gradient(1200px 600px at 50% -10%, #0f141a 0%, #0b0e12 55%, #070a0f 100%) !important;
  color: var(--text);
}
.kpi{ background: linear-gradient(180deg,#11161C,#0E1319); border:1px solid #1c232c; border-radius:16px; padding:14px 16px; box-shadow:0 10px 30px rgba(0,0,0,.25); }
.kpi .value{ font-size:22px; font-weight:700; color:var(--gold); }
.kpi .label{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }
.badge{ display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid #2a313a; background:#121822; color:#cdd6e2; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ===== Sidebar: carga de datos =====
st.sidebar.markdown("### 💼 Predicción BBVA & Santander")
uploaded = st.sidebar.file_uploader("📄 Subir CSV (date,ticker,y_true,y_pred)", type=["csv"])
range_default = (date(2025,11,1), date(2025,11,10))
d0, d1 = st.sidebar.date_input("Rango (principios de noviembre)", value=range_default)

if uploaded:
    df_all = pd.read_csv(uploaded)
    mode = "USUARIO (archivo subido)"
else:
    # Lee del repo si el nb 06 ya exportó
    local_path = "data/app/predicciones.csv"
    if os.path.exists(local_path):
        df_all = pd.read_csv(local_path)
        mode = "USUARIO (data/app/predicciones.csv)"
    else:
        st.error("No encontré datos. Sube un CSV o exporta desde el notebook 06 a data/app/predicciones.csv")
        st.stop()

# Normaliza
df_all.columns = [c.lower().strip() for c in df_all.columns]
df_all["date"] = pd.to_datetime(df_all["date"]).dt.date
df_all = df_all[["date","ticker","y_true","y_pred"]].copy()
df_all["y_true"] = pd.to_numeric(df_all["y_true"], errors="coerce")
df_all["y_pred"] = pd.to_numeric(df_all["y_pred"], errors="coerce")

# Filtro de rango
df_all = df_all[(df_all["date"]>=d0) & (df_all["date"]<=d1)]
valid_tickers = sorted(df_all["ticker"].dropna().unique().tolist())
if len(valid_tickers)==0:
    st.warning("No hay datos en el rango seleccionado.")
    st.stop()

# ===== Cabecera
left, right = st.columns([0.78, 0.22])
with left:
    st.markdown("<h1>📈 Plataforma de Predicción <span style='color:#CBA135'>BBVA</span> & <span style='color:#CBA135'>Santander</span></h1>", unsafe_allow_html=True)
    st.caption("Vista para inversores: compara predicciones vs. realidad, revisa métricas y exporta el informe.")
with right:
    st.markdown(f"<div style='text-align:right;margin-top:8px;'><span class='badge'>Modo: <b>{mode}</b></span></div>", unsafe_allow_html=True)

st.markdown("---")

# ===== KPIs
c1, c2 = st.columns(2)
for tk, col in zip(["BBVA","SAN"], [c1,c2]):
    dft = df_all[df_all["ticker"]==tk]
    if dft.empty: 
        continue
    m = metrics_segment(dft["y_true"].values, dft["y_pred"].values)
    with col:
        st.markdown(f"#### {tk}")
        k1,k2,k3 = st.columns(3)
        k1.markdown(f"<div class='kpi'><div class='label'>RMSE</div><div class='value'>{m['RMSE'] if not np.isnan(m['RMSE']) else '—':.4f}</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi'><div class='label'>MAE</div><div class='value'>{m['MAE'] if not np.isnan(m['MAE']) else '—':.4f}</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi'><div class='label'>MAPE</div><div class='value'>{m['MAPE'] if not np.isnan(m['MAPE']) else '—':.2f}%</div></div>", unsafe_allow_html=True)

# ===== Gráficos
st.markdown("### 📊 Series temporales")
if "BBVA" in valid_tickers:
    st.plotly_chart(plot_series(df_all[df_all["ticker"]=="BBVA"].sort_values("date"), "BBVA"), use_container_width=True)
if "SAN" in valid_tickers:
    st.plotly_chart(plot_series(df_all[df_all["ticker"]=="SAN"].sort_values("date"), "Santander"), use_container_width=True)

st.markdown("---")

# ===== Tabla y descarga
st.markdown("### 📑 Detalle")
st.dataframe(df_all.sort_values(["ticker","date"]).reset_index(drop=True), use_container_width=True, hide_index=True)

csv_bytes = df_all.sort_values(["ticker","date"]).to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar CSV filtrado", data=csv_bytes,
                   file_name=f"predicciones_{d0}_a_{d1}.csv", mime="text/csv")

st.caption("Las métricas solo se calculan en días con valor real (y_true). En días futuros aparece solo la predicción.")
