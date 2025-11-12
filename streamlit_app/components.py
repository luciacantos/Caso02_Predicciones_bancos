import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ===== Paleta consistente con la app =====
BLUE = "#1E90FF"      # real
LIGHT_BLUE = "#8cc4ff"  # predicción test
RED = "#FF4C4C"       # forecast futuro
TEXT = "#E6F1FF"
GRID = "#17324D"
PLOT_BG = "rgba(13,27,42,0.85)"  # azul oscuro translúcido

# ---------------- Métricas ----------------
def metrics_segment(y_true, y_pred):
    """RMSE/MAE/MAPE ignorando NaNs en y_true (solo en días con verdad)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
    mae  = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    denom = np.clip(np.abs(y_true[mask]), 1e-8, None)
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom)) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

# ---------------- Gráficas ----------------
def plot_series(df_ticker: pd.DataFrame, name: str) -> go.Figure:
    """
    Real (sólido azul), Predicción test (azul claro discontinua) y
    Forecast futuro (rojo con puntos) si hay días sin y_true.
    Espera columnas: date, y_true, y_pred.
    """
    df = df_ticker.sort_values("date").copy()
    fig = go.Figure()

    # Real (histórico con verdad)
    hist = df.dropna(subset=["y_true"])
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["y_true"],
            mode="lines",
            name=f"{name} · Real (últimos)",
            line=dict(width=2, color=BLUE)
        ))

    # Predicción sobre todo el rango
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["y_pred"],
        mode="lines+markers",
        name=f"{name} · Predicción test",
        line=dict(width=2, dash="dash", color=LIGHT_BLUE),
        marker=dict(size=5)
    ))

    # Forecast (días sin verdad)
    fc = df[df["y_true"].isna()]
    if not fc.empty:
        fig.add_trace(go.Scatter(
            x=fc["date"], y=fc["y_pred"],
            mode="markers+lines",
            name="Forecast (t+1…)",
            line=dict(width=0),
            marker=dict(size=7, color=RED)
        ))

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
        title=f"Evolución {name} — Real vs Predicción"
    )
    return fig

def plot_price(df_price: pd.DataFrame, title: str) -> go.Figure:
    """
    Línea simple para precio de cierre. Espera columnas: date, close.
    """
    d = df_price.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["close"],
        mode="lines",
        name="Precio cierre",
        line=dict(width=2, color=BLUE)
    ))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
        title=title
    )
    return fig

