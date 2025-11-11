import numpy as np
import pandas as pd
import plotly.graph_objects as go

def metrics_segment(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))
    mae  = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.clip(np.abs(y_true[mask]), 1e-8, None))) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def plot_series(df_ticker: pd.DataFrame, name: str) -> go.Figure:
    fig = go.Figure()
    hist = df_ticker.dropna(subset=["y_true"])
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["y_true"],
                                 mode="lines+markers", name=f"{name} · Real", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df_ticker["date"], y=df_ticker["y_pred"],
                             mode="lines+markers", name=f"{name} · Predicción", line=dict(width=3)))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,22,28,1)",
        font=dict(color="#EDEDED"), legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#2a313a"), yaxis=dict(gridcolor="#2a313a"),
        title=f"Evolución {name} — Real vs Predicción"
    )
    return fig
