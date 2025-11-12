import os
import base64
from datetime import date, datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =================== CONFIG ===================
st.set_page_config(page_title="Radar de Inversión — BBVA & Santander", page_icon="💹", layout="wide")

# ---------- Fondo con imagen local ----------
def set_background(image_path: str):
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{b64}") no-repeat center center fixed;
            background-size: cover;
        }}
        .stApp::before {{
            content:""; position:fixed; inset:0;
            background: radial-gradient(1400px 700px at 50% -10%, rgba(10,25,47,.92) 0%, rgba(13,27,42,.90) 55%, rgba(9,21,35,.92) 100%);
            pointer-events:none; z-index:-1;
        }}
        </style>
        """, unsafe_allow_html=True)
    except Exception:
        pass

set_background("streamlit_app/assets/bg.jpg")

# Paleta
BLUE = "#1E90FF"
LIGHT_BLUE = "#8cc4ff"
RED = "#FF4C4C"
TEXT = "#E6F1FF"
GRID = "#17324D"
PLOT_BG = "rgba(13,27,42,0.85)"

# =================== ESTILO EXTRA ===================
st.markdown(f"""
<style>
:root {{ --blue:{BLUE}; --red:{RED}; --text:{TEXT}; --grid:{GRID}; }}
h1,h2,h3 {{ color: var(--blue); letter-spacing:.3px; }}
hr {{ border:none; height:1px; background:linear-gradient(90deg,transparent,var(--grid),transparent); }}
.badge {{ display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid var(--grid);
         background:#0F2138; color:#B0C4DE; font-size:12px; }}
.kpi {{ background: linear-gradient(180deg, rgba(13,27,42,.95), #091523);
       border:1px solid var(--grid); border-radius:16px; padding:12px 16px; box-shadow:0 8px 24px rgba(0,0,0,.28); }}
.kpi .label {{ font-size:12px; color:#A9B0B8; text-transform:uppercase; letter-spacing:1px; }}
.kpi .value {{ font-size:22px; font-weight:700; color: var(--blue); }}
</style>
""", unsafe_allow_html=True)

# =================== UTILIDADES ===================
def fmt_num(v, fmt="{:.4f}"):
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "—"
    try:
        return fmt.format(v)
    except Exception:
        return str(v)

def metrics_segment(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))
    mae  = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    denom = np.clip(np.abs(y_true[mask]), 1e-8, None)
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom)) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def confidence_badge(mape: float) -> str:
    if isinstance(mape, (float, np.floating)) and np.isnan(mape):
        return "<span class='badge'>Confianza: —</span>"
    if mape <= 2.0:   lvl, color = "Alta",  "#21c55d"
    elif mape <= 5.0: lvl, color = "Media", "#f59e0b"
    else:             lvl, color = "Baja",  "#ef4444"
    return f"<span class='badge' style='border-color:{color}; color:{color};'>Confianza: {lvl} (MAPE {mape:.1f}%)</span>"

def recomendacion_simple(delta7: float, mape: float) -> str:
    if np.isnan(delta7) or np.isnan(mape):
        return "Sin recomendación (faltan datos)."
    conf = "alta" if mape <= 2 else ("media" if mape <= 5 else "baja")
    if delta7 >= 1.5 and mape <= 5:
        return f"Comprar ({delta7:.1f}%, confianza {conf})."
    if 0.5 <= delta7 < 1.5:
        return f"Mantener / compra parcial ({delta7:.1f}%, confianza {conf})."
    if -1.0 < delta7 < 0.5:
        return f"Mantener (variación {delta7:.1f}%)."
    return f"Vender / reducir ({delta7:.1f}%, confianza {conf})."

# =================== CARGA DE DATOS (LOCAL) ===================
def load_price_csv(path: str, ticker: str) -> pd.DataFrame:
    header_idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip().lower().startswith("date"):
                header_idx = i
                break

    colnames = ["date", "price", "adj_close", "close", "dividends",
                "high", "low", "open", "stock_splits", "volume"]

    df = pd.read_csv(path, header=None, names=colnames, skiprows=header_idx+1)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for c in ["price","adj_close","close","dividends","high","low","open","stock_splits","volume"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

    close_series = df["adj_close"] if df["adj_close"].notna().any() else df["close"]
    out = pd.DataFrame({"date": df["date"], "close": close_series}).dropna(subset=["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out["ticker"] = ticker
    return out

@st.cache_data(show_spinner=False)
def load_local_history():
    bbva_path = "data/BBVA.csv"
    san_path  = "data/SANTANDER.csv"
    if not os.path.exists(bbva_path) or not os.path.exists(san_path):
        st.error("⚠️ Faltan `data/BBVA.csv` o `data/SANTANDER.csv`.")
        st.stop()
    bbva = load_price_csv(bbva_path, "BBVA")
    san  = load_price_csv(san_path,  "SAN")
    return pd.concat([bbva, san], ignore_index=True)

df_hist = load_local_history()

# Predicciones locales
pred_path = "data/app/predicciones.csv"
if os.path.exists(pred_path):
    df_pred = pd.read_csv(pred_path)
    df_pred.columns = [c.lower().strip() for c in df_pred.columns]
    df_pred["date"] = pd.to_datetime(df_pred["date"], errors="coerce").dt.date
    df_pred["y_true"] = pd.to_numeric(df_pred["y_true"], errors="coerce")
    df_pred["y_pred"] = pd.to_numeric(df_pred["y_pred"], errors="coerce")
else:
    df_pred = pd.DataFrame(columns=["date","ticker","y_true","y_pred"])
    st.info("🔹 Aún no se ha generado `data/app/predicciones.csv`. Exporta desde el notebook 06 para habilitar la predicción.")

# =================== CABECERA ===================
c1, c2, c3 = st.columns([0.17, 0.66, 0.17])
with c1:
    try: st.image("streamlit_app/assets/bbva.png", use_container_width=True)
    except Exception: st.write("")
with c2:
    st.markdown(
        f"""
        <h1 style='text-align:center;'>💹 Radar de Inversión — <span style="color:{BLUE}">BBVA</span> & <span style="color:{RED}">Santander</span></h1>
        <p style='text-align:center; color:{TEXT};'>Histórico de 25 años y predicción con señales de inversión basadas en la confianza del modelo.</p>
        """, unsafe_allow_html=True
    )
with c3:
    try: st.image("streamlit_app/assets/santander.png", use_container_width=True)
    except Exception: st.write("")
st.markdown("<hr>", unsafe_allow_html=True)

# =================== HISTÓRICO (25 años) ===================
def plot_price(df_price: pd.DataFrame, title: str) -> go.Figure:
    d = df_price.dropna(subset=["close"]).sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["close"], mode="lines",
        name="Precio cierre", line=dict(width=2, color=BLUE)
    ))
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT), legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
        title=title
    )
    return fig

st.markdown("### 📈 Precio de cierre — 25 años")
cA, cB = st.columns(2)
with cA:
    st.plotly_chart(
        plot_price(df_hist[df_hist["ticker"]=="BBVA"], "BBVA — Precio de cierre (€)"),
        use_container_width=True
    )
with cB:
    st.plotly_chart(
        plot_price(df_hist[df_hist["ticker"]=="SAN"], "Santander — Precio de cierre (€)"),
        use_container_width=True
    )

st.markdown("---")

# =================== BOTÓN: MOSTRAR PREDICCIÓN ===================
if "show_pred" not in st.session_state:
    st.session_state.show_pred = False

if not st.session_state.show_pred:
    st.button("🔮 Mostrar predicción y recomendaciones", key="btn_pred",
              on_click=lambda: st.session_state.update(show_pred=True))
else:
    st.button("Ocultar predicción", key="btn_hide",
              on_click=lambda: st.session_state.update(show_pred=False))

# =================== PREDICCIÓN + KPIs + RECOMENDACIONES ===================
if st.session_state.show_pred:
    if df_pred.empty:
        st.warning("Aún no hay predicciones en `data/app/predicciones.csv`.")
    else:
        st.markdown("### 🔮 Predicción (test + forecast)")
        c1, c2 = st.columns(2)
        for tk, col in zip(["BBVA","SAN"], [c1,c2]):
            dtk = df_pred[df_pred["ticker"]==tk].sort_values("date")
            if dtk.empty:
                continue
            met = metrics_segment(dtk["y_true"].values, dtk["y_pred"].values)
            with col:
                st.markdown(f"#### {tk} {confidence_badge(met['MAPE'])}", unsafe_allow_html=True)
                k1,k2,k3 = st.columns(3)
                k1.markdown(f"<div class='kpi'><div class='label'>RMSE</div><div class='value'>{fmt_num(met['RMSE'])}</div></div>", unsafe_allow_html=True)
                k2.markdown(f"<div class='kpi'><div class='label'>MAE</div><div class='value'>{fmt_num(met['MAE'])}</div></div>", unsafe_allow_html=True)
                k3.markdown(f"<div class='kpi'><div class='label'>MAPE</div><div class='value'>{fmt_num(met['MAPE'], '{:.2f}%')}</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        def plot_pred(df_ticker: pd.DataFrame, name: str) -> go.Figure:
            fig = go.Figure()
            df_ticker = df_ticker.sort_values("date")
            hist = df_ticker.dropna(subset=["y_true"])
            if not hist.empty:
                fig.add_trace(go.Scatter(
                    x=hist["date"], y=hist["y_true"], mode="lines",
                    name="Real (últimos)", line=dict(width=2, color=BLUE)
                ))
            fig.add_trace(go.Scatter(
                x=df_ticker["date"], y=df_ticker["y_pred"], mode="lines+markers",
                name="Predicción test", line=dict(width=2, dash="dash", color=LIGHT_BLUE),
                marker=dict(size=5)
            ))
            fc = df_ticker[df_ticker["y_true"].isna()]
            if not fc.empty:
                fig.add_trace(go.Scatter(
                    x=fc["date"], y=fc["y_pred"], mode="markers+lines",
                    name="Forecast (t+1…)", line=dict(width=0),
                    marker=dict(size=7, color=RED)
                ))
            fig.update_layout(
                height=420, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG,
                font=dict(color=TEXT), legend=dict(bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
                title=f"{name} — Real vs Predicción"
            )
            return fig

        if "BBVA" in df_pred["ticker"].unique():
            st.plotly_chart(plot_pred(df_pred[df_pred["ticker"]=="BBVA"], "BBVA"), use_container_width=True)
        if "SAN" in df_pred["ticker"].unique():
            st.plotly_chart(plot_pred(df_pred[df_pred["ticker"]=="SAN"], "Santander"), use_container_width=True)

        st.markdown("### 🧭 Recomendaciones automáticas")
        recs = []
        for tk in ["BBVA","SAN"]:
            dtk = df_pred[df_pred["ticker"]==tk].sort_values("date")
            if dtk.empty or dtk["y_pred"].dropna().shape[0] < 2:
                recs.append((tk, "Sin datos suficientes.", np.nan, np.nan)); continue
            first = dtk["y_pred"].dropna().iloc[0]
            last  = dtk["y_pred"].dropna().iloc[-1]
            delta7 = (last-first)/first*100 if first else np.nan
            met = metrics_segment(dtk["y_true"].values, dtk["y_pred"].values)
            recs.append((tk, recomendacion_simple(delta7, met["MAPE"]), delta7, met["MAPE"]))
        df_recs = pd.DataFrame(recs, columns=["Ticker","Recomendación","Δ% prev. (≈rango)","MAPE"])
        st.dataframe(df_recs, use_container_width=True, hide_index=True)

st.caption(f"Datos cargados a las {datetime.now().strftime('%H:%M')}. Histórico: data/*.csv · Predicciones: data/app/predicciones.csv")
