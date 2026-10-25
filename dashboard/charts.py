"""Plotly chart constructors; no file I/O or experimental computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dashboard.styles import color_for

PLOT_LAYOUT = dict(template="plotly_dark", paper_bgcolor="#071522", plot_bgcolor="#0b1d2d",
                   font=dict(color="#dceaf1"), margin=dict(l=35, r=25, t=55, b=45),
                   legend=dict(orientation="h", y=-.22))


def finish(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    fig.update_layout(**PLOT_LAYOUT, title=title, height=height)
    fig.update_xaxes(gridcolor="#244257", zerolinecolor="#426174")
    fig.update_yaxes(gridcolor="#244257", zerolinecolor="#426174")
    return fig


def equity_chart(paths: pd.DataFrame, names: list[str]) -> go.Figure:
    fig = go.Figure()
    for name in names:
        frame = paths[paths.Strategy.eq(name)].sort_values("Date")
        fig.add_trace(go.Scatter(x=frame.Date, y=frame.Equity, name=name,
                                 line=dict(color=color_for(name), width=2)))
    fig.update_yaxes(title="Portfolio equity (INR)")
    return finish(fig, "Shared-capital equity paths · delivery costs")


def drawdown_chart(paths: pd.DataFrame, names: list[str]) -> go.Figure:
    fig = go.Figure()
    for name in names:
        frame = paths[paths.Strategy.eq(name)].sort_values("Date")
        equity = frame.Equity.to_numpy(float)
        start = equity[0] / (1 + frame.Daily_Return.iloc[0])
        full = np.r_[start, equity]
        dd = 100 * (full / np.maximum.accumulate(full) - 1)[1:]
        fig.add_trace(go.Scatter(x=frame.Date, y=dd, name=name,
                                 line=dict(color=color_for(name), width=2)))
    fig.update_yaxes(title="Drawdown (%)")
    return finish(fig, "Drawdown from running equity peak")


def strategy_bars(frame: pd.DataFrame, metric: str, title: str, names: list[str] | None = None) -> go.Figure:
    frame = frame[frame.Strategy.isin(names)].copy() if names is not None else frame.copy()
    fig = go.Figure(go.Bar(x=frame.Strategy, y=frame[metric], marker_color=[color_for(n) for n in frame.Strategy]))
    fig.update_yaxes(title=metric.replace("_", " "))
    fig.update_xaxes(tickangle=-35)
    return finish(fig, title, 360)


def window_heatmap(frame: pd.DataFrame, ml: list[str]) -> go.Figure:
    p = frame[frame.Strategy.isin(ml)].pivot(index="Strategy", columns=["Condition", "Window_Sessions"],
                                              values="Return_Diff_vs_Buy_Hold_Points")
    p = p.reindex(index=ml, columns=pd.MultiIndex.from_product([["worst", "best"], [100, 150, 200, 252, 378]]))
    z = p.to_numpy(float)
    limit = max(abs(np.nanmin(z)), abs(np.nanmax(z)))
    fig = go.Figure(go.Heatmap(z=z, x=[f"{a} {b}" for a, b in p.columns], y=p.index,
                               colorscale="RdBu", zmin=-limit, zmax=limit,
                               colorbar=dict(title="pp"), hovertemplate="%{y}<br>%{x}<br>%{z:.2f} pp<extra></extra>"))
    return finish(fig, "ML return minus Buy & Hold · percentage points", 450)


def window_equity(paths: pd.DataFrame, names: list[str], start: str, end: str) -> tuple[go.Figure, go.Figure]:
    line = go.Figure(); draw = go.Figure()
    for name in names:
        sub = paths[paths.Strategy.eq(name)].sort_values("Date").reset_index(drop=True)
        selected = sub[(sub.Date >= start) & (sub.Date <= end)]
        if selected.empty:
            continue
        first = selected.index[0]
        pre = float(sub.Equity.iloc[first - 1]) if first else float(sub.Equity.iloc[0] / (1 + sub.Daily_Return.iloc[0]))
        equity = np.r_[pre, selected.Equity.to_numpy(float)]
        prior_date = sub.Date.iloc[first - 1] if first else selected.Date.iloc[0] - pd.Timedelta(days=1)
        x = [prior_date] + selected.Date.tolist()
        line.add_trace(go.Scatter(x=x, y=100 * equity / pre, name=name,
                                  line=dict(color=color_for(name))))
        draw.add_trace(go.Scatter(x=x, y=100 * (equity / np.maximum.accumulate(equity) - 1),
                                  name=name, line=dict(color=color_for(name))))
    line.update_yaxes(title="Equity index (pre-window=100)")
    draw.update_yaxes(title="Window drawdown (%)")
    return finish(line, "Continuous portfolio equity slice"), finish(draw, "Drawdown within selected slice")


def forest(frame: pd.DataFrame) -> go.Figure:
    ordered = frame.sort_values("Annualized_Log_Excess_Pct")
    y = ordered.Policy.tolist()
    x = ordered.Annualized_Log_Excess_Pct.to_numpy(float)
    low = x - ordered.CI_95_Lower_Pct.to_numpy(float)
    high = ordered.CI_95_Upper_Pct.to_numpy(float) - x
    color = ["#e4a563" if p < .05 else "#23b4c5" for p in ordered.Holm_P]
    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=10, color=color),
                               error_x=dict(type="data", symmetric=False, array=high, arrayminus=low),
                               customdata=ordered.Holm_P,
                               hovertemplate="%{y}<br>Excess: %{x:.2f}%<br>Holm p: %{customdata:.3f}<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color="#98acb8")
    fig.update_xaxes(title="Annualized mean daily log-return difference vs Buy & Hold (%)")
    return finish(fig, "Paired 20-session block-bootstrap intervals · delivery costs", 600)
