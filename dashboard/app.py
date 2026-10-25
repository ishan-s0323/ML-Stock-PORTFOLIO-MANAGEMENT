"""Interactive, read-only academic dashboard for the saved yfinance rebuild."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc

from dashboard import charts
from dashboard.data_loader import (BASE, DATA, ML, ROOT as DATA_ROOT, DataUnavailable,
                                   load_manifest, load_stock, load_table, source_label,
                                   strategies, symbols, validate_selected_windows)
from dashboard.styles import CSS, color_for

st.set_page_config(page_title="ML Stock Portfolio Management", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

PAGES = ["Research Overview", "Stock Data Explorer", "ML Model Performance",
         "Shared-Capital Backtesting", "Transaction Costs", "Best vs Worst Windows",
         "Bullish vs Bearish Regimes", "Exposure-Matched Analysis",
         "ML Interpretability", "Statistical Validation", "Methodology & Limitations"]
DEMO = ["Research Overview", "ML Model Performance", "Shared-Capital Backtesting",
        "Transaction Costs", "Best vs Worst Windows", "Best vs Worst Windows",
        "Exposure-Matched Analysis", "ML Interpretability", "Methodology & Limitations"]
DEMO_LABELS = ["1 Overview", "2 Models", "3 Full-period portfolio", "4 Transaction costs",
               "5 Worst window", "6 Best window", "7 Exposure explanation",
               "8 Interpretability", "9 Conclusions"]


@st.cache_data(show_spinner=False)
def table(key: str) -> pd.DataFrame:
    frame = load_table(key)
    if key == "portfolio_paths":
        frame["Date"] = pd.to_datetime(frame.Date)
    return frame


@st.cache_data(show_spinner=False)
def manifest() -> dict:
    return load_manifest()


@st.cache_data(show_spinner=False)
def stock_data(symbol: str) -> pd.DataFrame:
    return load_stock(symbol)


def header(title: str, subtitle: str = "") -> None:
    st.markdown('<div class="eyebrow">Research status: Exploratory — Historical Simulation</div>', unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f'<div class="subtle">{subtitle}</div>', unsafe_allow_html=True)


def note(text: str, warning: bool = False) -> None:
    kind = "warning-note" if warning else "note"
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


def source(key: str) -> None:
    st.caption(f"Saved source: {source_label(key)} · rebuilt yfinance panel")


def download(frame: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), filename,
                       mime="text/csv", key=f"download_{filename}")


def show_plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def overview() -> None:
    header("ML Stock Portfolio Management", "Cost-Aware Machine Learning Trading Signals Across 50 Indian Stocks")
    note("Academic research dashboard · historical simulation · not investment advice.", True)
    m = manifest(); summary = table("portfolio_summary")
    picks = {name: summary.set_index("Strategy").loc[name] for name in
             ["Buy & Hold", "Logistic Regression 1D — 2021 selected", "XGBoost 1D"]}
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Buy & Hold · delivery", f"{picks['Buy & Hold'].Total_Return_Pct:+.2f}%")
    k2.metric("Selected logistic · delivery", f"{picks['Logistic Regression 1D — 2021 selected'].Total_Return_Pct:+.2f}%")
    k3.metric("Fixed XGBoost 1D · delivery", f"{picks['XGBoost 1D'].Total_Return_Pct:+.2f}%")
    k4.metric("Passive max drawdown", f"{picks['Buy & Hold'].Max_Drawdown_Pct:.2f}%")
    a, b = st.columns([1.7, 1])
    with a:
        st.subheader("Research pipeline")
        st.markdown("**Yahoo Finance** → **17 features** → **chronological ML training** → **signals** → **shared-capital simulation** → **statistical evaluation**")
        st.subheader("Research questions")
        st.markdown("1. How do separately funded stock averages differ from shared-capital outcomes?\n"
                    "2. How do transaction costs, horizons, and threshold rules affect turnover and returns?\n"
                    "3. How do ML policies compare with passive holding under costs, exposure controls, and different market windows?")
        note("The full-period passive portfolio leads under delivery costs. In hindsight-selected weak windows, ML often loses less; in strong windows it gives up upside. These are descriptive results from an inspected sample.")
    with b:
        st.subheader("Verified experiment")
        st.write(f"**Stocks:** {len(m['symbols'])} Indian equities")
        st.write(f"**Data:** {m['start_inclusive']} to 2025-12-30")
        st.write("**Evaluation:** 2022–2025 · 986 execution sessions")
        st.write("**Models:** XGBoost 1D, 3D, 5D; logistic regression 1D")
        st.write("**Evaluation folds:** 16 quarterly folds")
        st.write(f"**Starting shared capital:** INR {picks['Buy & Hold'].Initial_Capital:,.0f}")
        st.write("**Universe:** fixed retrospective 50-name set")
    source("portfolio_summary")


def stock_explorer() -> None:
    header("Stock Data Explorer", "Adjusted historical OHLCV and saved technical indicators")
    if not (DATA / "processed").is_dir():
        st.info("Processed Yahoo price CSVs are intentionally excluded from the public GitHub package. "
                "For the complete local explorer, restore the matching manifest and 50 processed files to "
                "`data/yfinance_rebuild/` and run the dashboard from the repository root.")
        return
    names = symbols()
    symbol = st.selectbox("Stock", names, index=names.index("RELIANCE") if "RELIANCE" in names else 0)
    frame = stock_data(symbol)
    selected_dates = st.date_input("Date range", value=(frame.Date.iloc[-252].date(), frame.Date.iloc[-1].date()),
                                   min_value=frame.Date.iloc[0].date(), max_value=frame.Date.iloc[-1].date())
    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Choose both a start and end date.")
        return
    start, end = selected_dates
    sub = frame[(frame.Date.dt.date >= start) & (frame.Date.dt.date <= end)]
    if sub.empty:
        st.info("No saved sessions in that date range.")
        return
    st.caption(f"{symbol}.NS · {len(sub):,} sessions shown · source: documented yfinance acquisition")
    overlays = st.multiselect("Overlays", ["SMA 20", "SMA 50", "SMA 200", "Bollinger Bands"],
                              default=["SMA 20", "SMA 50"])
    fig = go.Figure(go.Candlestick(x=sub.Date, open=sub.Open, high=sub.High, low=sub.Low, close=sub.Close,
                                   name="Adjusted OHLC"))
    for label, col, color in [("SMA 20", "SMA_20", "#18b6c9"), ("SMA 50", "SMA_50", "#edb35a"),
                              ("SMA 200", "SMA_200", "#ad91e6")]:
        if label in overlays:
            fig.add_trace(go.Scatter(x=sub.Date, y=sub[col], name=label, line=dict(color=color, width=1.4)))
    if "Bollinger Bands" in overlays:
        for col in ("BBU_20_2.0", "BBL_20_2.0"):
            fig.add_trace(go.Scatter(x=sub.Date, y=sub[col], name=col, line=dict(color="#9badb8", dash="dot")))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title="Adjusted price (INR)")
    show_plot(charts.finish(fig, f"{symbol} adjusted OHLCV", 500))
    volume = go.Figure(go.Bar(x=sub.Date, y=sub.Volume, marker_color="#18b6c9"))
    volume.update_yaxes(title="Raw volume")
    show_plot(charts.finish(volume, "Volume", 230))
    c1, c2 = st.columns(2)
    with c1:
        rsi = go.Figure(go.Scatter(x=sub.Date, y=sub.RSI_14, line=dict(color="#edb35a")))
        rsi.add_hline(y=70, line_dash="dash", line_color="#aebac2")
        rsi.add_hline(y=30, line_dash="dash", line_color="#aebac2")
        show_plot(charts.finish(rsi, "RSI (14)", 280))
    with c2:
        macd = go.Figure()
        macd.add_trace(go.Scatter(x=sub.Date, y=sub.MACD_12_26_9, name="MACD", line=dict(color="#18b6c9")))
        macd.add_trace(go.Scatter(x=sub.Date, y=sub.MACDs_12_26_9, name="Signal", line=dict(color="#edb35a")))
        show_plot(charts.finish(macd, "MACD (12,26,9)", 280))
    note("OHLC fields are back-adjusted using each day's Adj Close / raw Close factor. Raw volume is retained. Back-adjusted prices support consistent analysis but are not executable historical quotes.")
    download(sub, "Download displayed stock rows", f"{symbol}_adjusted_display.csv")


def model_performance() -> None:
    header("ML Model Performance", "Quarterly walk-forward classification diagnostics")
    summary = table("model_summary"); folds = table("folds")
    chosen = st.multiselect("Models", summary.Model.tolist(), default=summary.Model.tolist())
    sub = summary[summary.Model.isin(chosen)]
    if sub.empty:
        st.info("Select at least one model."); return
    cols = st.columns(3)
    for col, metric, label in zip(cols, ["Mean_Fold_AUC", "Mean_Fold_Accuracy", "Mean_Fold_LogLoss"],
                                  ["Mean fold ROC-AUC", "Mean fold accuracy", "Mean fold log loss"]):
        fig = go.Figure(go.Bar(x=sub.Model, y=sub[metric], marker_color=[color_for(n) for n in sub.Model]))
        fig.update_xaxes(tickangle=-25)
        with col: show_plot(charts.finish(fig, label, 310))
    st.dataframe(sub, width="stretch", hide_index=True)
    fold = folds[folds.Model.isin(chosen)].copy()
    stocks = sorted(fold.Stock.unique())
    pick = st.selectbox("Stock-level fold diagnostics", ["All stocks"] + stocks)
    if pick != "All stocks": fold = fold[fold.Stock.eq(pick)]
    trend = fold.groupby(["Fold", "Model"], as_index=False)[["AUC", "Accuracy", "LogLoss"]].mean(numeric_only=True)
    metric = st.selectbox("Quarterly metric", ["AUC", "Accuracy", "LogLoss"])
    fig = go.Figure()
    for model in chosen:
        one = trend[trend.Model.eq(model)]
        fig.add_trace(go.Scatter(x=one.Fold, y=one[metric], name=model,
                                 line=dict(color=color_for(model)), mode="lines+markers"))
    fig.update_xaxes(title="Walk-forward fold (1–16)")
    show_plot(charts.finish(fig, f"Quarterly {metric} · {pick}"))
    st.subheader("2021 diagnostic probabilities")
    note("These saved probabilities and labels are from the already-used 2021 validation year. They can support a real ROC curve and confusion matrix but are not a new holdout result.", True)
    probabilities = table("validation_probabilities")
    model = st.selectbox("Diagnostic model", chosen)
    stock_choice = st.selectbox("Diagnostic stock", ["All stocks"] + sorted(probabilities.Stock.unique()))
    p = probabilities[probabilities.Model.eq(model)]
    if stock_choice != "All stocks": p = p[p.Stock.eq(stock_choice)]
    threshold = st.slider("Probability cutoff for confusion matrix", 0.05, 0.95, 0.50, 0.05)
    if p.Outcome.nunique() == 2:
        matrix = confusion_matrix(p.Outcome, (p.Probability >= threshold).astype(int), labels=[0, 1])
        fpr, tpr, _ = roc_curve(p.Outcome, p.Probability)
        left, right = st.columns(2)
        with left:
            heat = go.Figure(go.Heatmap(z=matrix, x=["Predicted 0", "Predicted 1"], y=["Actual 0", "Actual 1"],
                                        colorscale="Teal", text=matrix, texttemplate="%{text}"))
            show_plot(charts.finish(heat, f"Observed confusion matrix · cutoff {threshold:.2f}", 330))
        with right:
            curve = go.Figure()
            curve.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC-AUC {auc(fpr, tpr):.3f}", line=dict(color="#18b6c9")))
            curve.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(dash="dash", color="#9baeb9")))
            curve.update_xaxes(title="False-positive rate"); curve.update_yaxes(title="True-positive rate")
            show_plot(charts.finish(curve, "ROC curve from saved labels and probabilities", 330))
    else:
        st.info("Both outcome classes are needed for this stock's ROC and confusion matrix.")
    download(fold, "Download displayed fold diagnostics", "model_fold_diagnostics.csv")
    note("Classification quality does not by itself imply positive trading returns after costs and capital constraints.")


def portfolio_page() -> None:
    header("Shared-Capital Portfolio Backtesting", "All policies compete for one INR 5 million cash account")
    summary = table("portfolio_summary"); risk = table("risk"); paths = table("portfolio_paths")
    names = strategies()
    chosen = st.multiselect("Compare strategies", names,
                            default=["Buy & Hold", "Logistic Regression 1D — 2021 selected", "XGBoost 1D", "SMA 200"])
    if not chosen: st.info("Select at least one policy."); return
    display = summary[summary.Strategy.isin(chosen)].merge(risk[["Strategy", "Sortino"]], on="Strategy", how="left")
    fields = ["Strategy", "Total_Return_Pct", "Sharpe", "Sortino", "Max_Drawdown_Pct",
              "Mean_Exposure_Pct", "Trade_Legs", "Fees_Paid"]
    st.dataframe(display[fields].round(3), width="stretch", hide_index=True)
    show_plot(charts.equity_chart(paths, chosen)); show_plot(charts.drawdown_chart(paths, chosen))
    st.subheader("Stock averages versus shared capital")
    gap = table("accounting_gap")
    st.dataframe(gap[gap.Policy.isin(chosen)].round(2), width="stretch", hide_index=True)
    note("Separately funded stock means assign fresh capital to every name. The shared portfolio limits entry tickets and allocates simultaneous orders from one cash balance, so returns and rankings can differ.")
    download(display[fields], "Download displayed portfolio metrics", "portfolio_comparison.csv")
    source("portfolio_summary")


def costs_page() -> None:
    header("Transaction Cost Analysis", "Saved scenarios only · no interpolation or live simulation")
    flat = table("cost_sweep"); delivery = table("portfolio_summary")
    names = strategies()
    chosen = st.multiselect("Strategies", names, default=["Buy & Hold", "XGBoost 1D", "XGBoost 3D",
                                                        "Logistic Regression 1D — 2021 selected"])
    if not chosen: st.info("Select at least one policy."); return
    subset = flat[flat.Strategy.isin(chosen)].copy()
    subset["Cost_bp_per_leg"] = subset.All_In_Cost_Per_Leg * 10000
    fig = go.Figure()
    for name in chosen:
        one = subset[subset.Strategy.eq(name)].sort_values("Cost_bp_per_leg")
        fig.add_trace(go.Scatter(x=one.Cost_bp_per_leg, y=one.Total_Return_Pct,
                                 mode="lines+markers", name=name, line=dict(color=color_for(name))))
    fig.update_xaxes(title="Flat cost per executed leg (bp)", tickvals=[0, 10, 20, 40])
    fig.update_yaxes(title="Cumulative return (%)")
    show_plot(charts.finish(fig, "Flat-cost sensitivity · shared portfolio", 360))
    st.subheader("Delivery-cost profile")
    prof = delivery[delivery.Strategy.isin(chosen)][["Strategy", "Total_Return_Pct", "Trade_Legs", "Fees_Paid", "Mean_Exposure_Pct"]]
    st.dataframe(prof.round(2), width="stretch", hide_index=True)
    show_plot(charts.strategy_bars(prof, "Total_Return_Pct", "Saved Indian delivery-cost scenario", chosen))
    fixed = flat[flat.Strategy.eq("XGBoost 1D")].set_index("All_In_Cost_Per_Leg")
    if set([0.0, .004]).issubset(fixed.index):
        note(f"Fixed XGBoost 1D falls from {fixed.loc[0, 'Total_Return_Pct']:+.2f}% at 0 bp to "
             f"{fixed.loc[.004, 'Total_Return_Pct']:+.2f}% at 40 bp. Zero-cost returns do not establish predictive skill.")
    note("The delivery profile applies illustrative Indian charges and 5 bp adverse slippage per executed leg. The flat-bp sweep is a separate legacy cost assumption; its paths were not recomputed under the delivery profile.", True)
    download(subset, "Download flat-cost rows", "flat_cost_sensitivity.csv")


def windows_page(demo_condition: str | None = None) -> None:
    header("Best vs Worst Market Windows", "Benchmark-selected hindsight cases · delivery-cost shared portfolio")
    windows = table("selected_windows"); validate_selected_windows(windows)
    summary = table("window_summary"); paths = table("portfolio_paths")
    c1, c2, c3 = st.columns(3)
    with c1: length = st.selectbox("Window length (sessions)", [100, 150, 200, 252, 378])
    with c2: condition = st.selectbox("Condition", ["worst", "best"], index=1 if demo_condition == "best" else 0)
    with c3: metric = st.selectbox("Comparison metric", ["Return_Pct", "Return_Diff_vs_Buy_Hold_Points",
                                                        "Max_Drawdown_Pct", "Sharpe", "Mean_Exposure_Pct",
                                                        "Trade_Legs", "Fee_Charges_INR"])
    subset = windows[(windows.Window_Sessions == length) & windows.Condition.eq(condition)]
    passive = subset[subset.Strategy.eq("Buy & Hold")].iloc[0]
    count = int(summary[(summary.Window_Sessions == length) & summary.Condition.eq(condition)].ML_Beating_Buy_Hold_Count.iloc[0])
    a, b, c, d = st.columns(4)
    a.metric("Start", str(passive.Start_Date)); b.metric("End", str(passive.End_Date))
    c.metric("Buy & Hold return", f"{passive.Return_Pct:+.2f}%")
    d.metric("ML beating passive", f"{count}/8")
    chosen = st.multiselect("ML strategies", ML, default=ML)
    plotted = subset[subset.Strategy.isin(["Buy & Hold"] + chosen)]
    show_plot(charts.strategy_bars(plotted, metric, f"{length}-session {condition} window · {metric}"))
    st.dataframe(plotted[["Strategy", "Return_Pct", "Return_Diff_vs_Buy_Hold_Points", "Max_Drawdown_Pct",
                          "Sharpe", "Mean_Exposure_Pct", "Trade_Legs", "Fee_Charges_INR"]].round(2),
                 width="stretch", hide_index=True)
    st.subheader("Across all five predefined lengths")
    show_plot(charts.window_heatmap(windows, ML))
    left, right = st.columns(2)
    for col, case in [(left, "best"), (right, "worst")]:
        one = windows[windows.Condition.eq(case)]
        fig = go.Figure()
        for name in ["Buy & Hold"] + ML:
            g = one[one.Strategy.eq(name)].sort_values("Window_Sessions")
            fig.add_trace(go.Scatter(x=g.Window_Sessions, y=g.Return_Pct, mode="lines+markers", name=name,
                                     line=dict(color=color_for(name), width=2.5 if name == "Buy & Hold" else 1.4)))
        fig.update_yaxes(title="Cumulative return (%)"); fig.update_xaxes(title="Sessions")
        with col: show_plot(charts.finish(fig, f"{case.title()} benchmark windows", 420))
    focus = ["Buy & Hold"] + chosen[:3]
    if len(focus) > 1:
        eq, dd = charts.window_equity(paths, focus, str(passive.Start_Date), str(passive.End_Date))
        l, r = st.columns(2)
        with l: show_plot(eq)
        with r: show_plot(dd)
    note("Windows were selected after observing Buy & Hold returns, then all eight ML policies were evaluated on those same dates. They overlap and are not independent evidence of timing skill. The worst 378-session window has a positive passive return and is not a bearish period.", True)
    download(plotted, "Download selected window rows", f"window_{length}_{condition}.csv")


def regimes_page() -> None:
    header("Bullish vs Bearish Market Regimes", "Prior-close 200-session moving-average labels · separate from selected windows")
    all_states = table("regime_all")
    state = st.selectbox("Regime", ["bullish", "bearish", "neutral"])
    selected = all_states[all_states.Regime.eq(state) & all_states.Strategy.isin(["Buy & Hold"] + ML)].copy()
    passive = selected[selected.Strategy.eq("Buy & Hold")].iloc[0]
    selected["Mean_Daily_Gap_vs_Buy_Hold_Points"] = selected.Mean_Daily_Return_Pct - passive.Mean_Daily_Return_Pct
    a, b, c = st.columns(3)
    a.metric("Sessions", int(passive.Sessions)); b.metric("Passive mean daily return", f"{passive.Mean_Daily_Return_Pct:+.3f}%")
    c.metric("Passive mean exposure", f"{passive.Mean_Exposure_Pct:.1f}%")
    l, r = st.columns(2)
    with l: show_plot(charts.strategy_bars(selected, "Mean_Daily_Return_Pct", f"{state.title()} · mean daily return"))
    with r: show_plot(charts.strategy_bars(selected, "Mean_Exposure_Pct", f"{state.title()} · invested exposure"))
    st.dataframe(selected[["Strategy", "Mean_Daily_Return_Pct", "Mean_Daily_Gap_vs_Buy_Hold_Points",
                          "Mean_Exposure_Pct", "Volatility_Annual_Pct", "Sharpe_On_Labelled_Days",
                          "Trade_Legs"]].round(3), width="stretch", hide_index=True)
    episodes = table("regime_episodes")
    ep = episodes[episodes.Regime.eq(state) & episodes.Strategy.isin(["Buy & Hold"] + ML)]
    st.subheader("Contiguous episodes")
    st.dataframe(ep[["Episode", "Start_Date", "End_Date", "Sessions", "Strategy", "Return_Pct",
                     "Max_Drawdown_Pct", "Mean_Exposure_Pct"]].round(2), width="stretch", hide_index=True)
    note("The state for session t uses only the internal passive portfolio through t−1. A bearish label does not mean session t fell. Returns compounded across noncontiguous labelled days are descriptive, not an executable regime-only strategy.", True)
    download(selected, "Download displayed regime metrics", f"regime_{state}.csv")


def matched_page() -> None:
    header("Exposure-Matched Analysis", "Retrospective passive/cash diagnostic · secondary comparison")
    matched = table("matched_windows"); direct = table("selected_windows")
    c1, c2, c3 = st.columns(3)
    with c1: length = st.selectbox("Sessions", [100, 150, 200, 252, 378])
    with c2: condition = st.selectbox("Window", ["worst", "best"])
    with c3: name = st.selectbox("ML policy", ML)
    row = matched[(matched.Window_Sessions == length) & matched.Condition.eq(condition) & matched.Strategy.eq(name)].iloc[0]
    passive = direct[(direct.Window_Sessions == length) & direct.Condition.eq(condition) & direct.Strategy.eq("Buy & Hold")].iloc[0]
    a, b, c = st.columns(3)
    a.metric("ML return", f"{row.Return_Pct:+.2f}%")
    b.metric("Buy & Hold", f"{passive.Return_Pct:+.2f}%")
    c.metric("Exposure-matched", f"{row.Matched_Return_Pct:+.2f}%")
    bar = pd.DataFrame({"Comparator": [name, "Buy & Hold", "Matched passive/cash"],
                        "Return_Pct": [row.Return_Pct, passive.Return_Pct, row.Matched_Return_Pct]})
    fig = go.Figure(go.Bar(x=bar.Comparator, y=bar.Return_Pct,
                           marker_color=[color_for(name), color_for("Buy & Hold"), "#a9b4c4"]))
    show_plot(charts.finish(fig, f"{condition.title()} {length}-session window · cumulative return"))
    st.write(f"**ML minus full passive:** {row.Return_Pct - passive.Return_Pct:+.2f} percentage points  ·  "
             f"**ML minus exposure match:** {row.Excess_vs_Matched_Points:+.2f} points")
    st.write(f"**Mean invested exposure:** ML {row.Mean_Exposure_Pct:.2f}% · matched {row.Matched_Exposure_Pct:.2f}% · "
             f"full passive {passive.Mean_Exposure_Pct:.2f}%")
    st.write(f"**Maximum drawdown:** ML {row.Max_Drawdown_Pct:.2f}% · match {row.Matched_Max_Drawdown_Pct:.2f}% · "
             f"full passive {passive.Max_Drawdown_Pct:.2f}%")
    note("The passive/cash weight is chosen using realized exposure from this same historical window. It is not a prospectively executable benchmark. Matching average exposure does not match beta, volatility, stock selection, or trading turnover.", True)
    download(matched[(matched.Window_Sessions == length) & matched.Condition.eq(condition)],
             "Download matched comparisons", f"matched_{length}_{condition}.csv")


def interpretability_page() -> None:
    header("ML Interpretability", "Exploratory diagnostics on the already-used 2021 validation year")
    importance = table("importance"); ablation = table("ablation")
    calibration = table("calibration"); bins = table("calibration_bins")
    model = st.selectbox("Model", importance.Model.unique().tolist())
    imp = importance[importance.Model.eq(model)].sort_values("Normalized_Importance", ascending=False)
    top = st.slider("Top features", 5, 17, 10)
    fig = go.Figure(go.Bar(x=imp.head(top).Normalized_Importance[::-1], y=imp.head(top).Feature[::-1],
                           orientation="h", marker_color=color_for(model)))
    fig.update_xaxes(title="Mean normalized importance")
    show_plot(charts.finish(fig, f"{model} feature importance", 420))
    st.caption("XGBoost: normalized gain. Logistic regression: absolute standardized coefficient importance. Association is not causality.")
    st.subheader("Group ablation")
    ab = ablation[ablation.Model.eq(model)]
    col1, col2 = st.columns(2)
    for col, field, title in [(col1, "AUC_Drop", "AUC drop after removal"),
                              (col2, "Brier_Increase", "Brier increase after removal")]:
        fig = go.Figure(go.Bar(x=ab.Removed_Group, y=ab[field], marker_color=color_for(model)))
        with col: show_plot(charts.finish(fig, title, 320))
    st.subheader("Probability calibration")
    cal = calibration[calibration.Model.eq(model)].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Pooled AUC", f"{cal.Pooled_AUC:.3f}")
    b.metric("Brier score", f"{cal.Pooled_Brier:.3f}")
    c.metric("Expected calibration error", f"{cal.Pooled_ECE:.3f}")
    one = bins[bins.Model.eq(model)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=one.Mean_Probability, y=one.Observed_Frequency,
                             mode="markers+lines", name=model, marker=dict(size=np.clip(one.Count / one.Count.max() * 20, 7, 20), color=color_for(model))))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", line=dict(dash="dash", color="#a0b3bd")))
    fig.update_xaxes(title="Mean predicted probability", range=[0, 1])
    fig.update_yaxes(title="Observed positive frequency", range=[0, 1])
    show_plot(charts.finish(fig, "Reliability diagram · saved 2021 probabilities"))
    note("Importance, ablation, and calibration describe fitted models on a validation year already used during development. They do not establish causal features or independently validated trading improvements.", True)
    download(imp, "Download feature rankings", f"importance_{model.replace(' ', '_')}.csv")


def stats_page() -> None:
    header("Statistical Validation", "Full 2022–2025 delivery-cost inference · separate from selected windows")
    boot = table("bootstrap"); sharpe = table("sharpe_bootstrap")
    placebo = table("placebo")
    show_plot(charts.forest(boot))
    st.dataframe(boot[["Policy", "Annualized_Log_Excess_Pct", "CI_95_Lower_Pct", "CI_95_Upper_Pct",
                       "Unadjusted_P", "Holm_P"]].round(3), width="stretch", hide_index=True)
    sig = boot[boot.Holm_P < .05]
    if not sig.empty:
        st.write("**Holm-significant log-return comparison(s):** " + ", ".join(sig.Policy))
    note("The paired circular moving-block bootstrap resamples the same market dates across policies (5,000 draws, 20-session blocks). Holm adjusts across 12 active policies. A 95% interval crossing zero does not support a difference at the stated level. The development period was repeatedly inspected, so even these adjusted p-values are exploratory.")
    st.subheader("Exposure-matched Sharpe differences")
    st.dataframe(sharpe[["Strategy", "Observed_Sharpe_Difference", "CI_2_5", "CI_97_5", "Raw_P", "Holm_Adjusted_P"]].round(3),
                 width="stretch", hide_index=True)
    st.subheader("Signal-rotation placebo · legacy flat 10 bp")
    st.dataframe(placebo.round(3), width="stretch", hide_index=True)
    note("The placebo uses legacy flat-10-bp paths. It was not repeated for the delivery-cost profile. Best/worst windows were selected retrospectively; this dashboard assigns them no new p-values.", True)
    download(boot, "Download bootstrap table", "delivery_bootstrap.csv")


def methodology_page() -> None:
    header("Methodology, Limitations & Reproducibility", "What the application reads and what the simulation assumes")
    m = manifest()
    diag_path = BASE / "feature_diagnostics/diagnostic_manifest.json"
    diag = __import__("json").loads(diag_path.read_text(encoding="utf-8")) if diag_path.is_file() else {}
    with st.expander("Acquisition and adjustment", expanded=True):
        st.write(f"**Provider:** {m['vendor']} · **version:** {m['yfinance_version']} · **retrieval:** {m['generated_utc']}")
        st.write(f"**Request:** {m['start_inclusive']} to exclusive {m['end_exclusive']}; {m['download_arguments']}")
        st.write("**Adjustment:** raw OHLC multiplied by same-day Adj Close / raw Close; raw volume retained. The manifest records hashes and corporate-action event counts.")
        st.write("**Universe:** fixed retrospective 50-name set, not point-in-time Nifty membership.")
    with st.expander("Features and fitted model settings"):
        features = diag.get("features", [])
        st.write(f"**Features ({len(features)}):** {', '.join(features)}")
        st.json(diag.get("xgboost_parameters", {}))
        st.write("Logistic regression: C=0.1, max_iter=500, random_state=42; training-only standardization. XGBoost: 100 trees, depth 3, learning rate 0.03, subsample and column sample 0.8.")
    with st.expander("Evaluation and execution"):
        st.write("Sixteen quarterly chronological folds evaluate 2022–2025. Inputs at close t produce next-open trades; 1-, 3-, and 5-session models enforce corresponding minimum holds. The shared INR 5 million account uses whole shares and INR 100,000 maximum entry tickets, with simultaneous requests constrained by available cash. Idle cash accrues an assumed 7% annually.")
        st.write("The delivery-cost profile applies illustrative Indian charges and 5 bp adverse slippage per leg. The flat 0/10/20/40 bp sweep is a separate legacy sensitivity. The simulator assumes full next-open execution on back-adjusted prices and liquidates at the final close.")
    with st.expander("Statistical and regime design"):
        st.write("Full-period comparisons use paired circular 20-session blocks, 5,000 replicates, common date indices, and Holm correction across 12 active policies. Best/worst windows use benchmark-only extrema for 100/150/200/252/378 sessions; they overlap and were selected with hindsight. Prior-close bullish/bearish labels use the previous close relative to a trailing 200-session moving average of the internal passive portfolio.")
    st.subheader("Limitations")
    st.markdown("- The 2022–2025 evaluation years were inspected repeatedly; they are not an untouched holdout.\n"
                "- The 50 stocks are a fixed retrospective universe with unverified historical index membership.\n"
                "- Back-adjusted Yahoo prices drive whole-share accounting but are not executable historical prices.\n"
                "- Some informal candidate trials were not logged; the reported multiple-testing family understates the full search.\n"
                "- Delivery charges and slippage are illustrative; impact, partial fills and settlement frictions are omitted.\n"
                "- Exposure matching uses realized exposure and does not match beta or stock composition.\n"
                "- Hindsight-selected, overlapping windows do not validate market-timing skill.")
    st.subheader("Documents and offline preparation")
    report = ROOT / "PROJECT_RESEARCH_REPORT_REGIME_V3.pdf"
    if report.is_file():
        st.download_button("Download latest IEEE research report", report.read_bytes(), report.name,
                           mime="application/pdf")
    for rel in ("README.md", "REGIME_ANALYSIS.md", "REGIME_DIRECT_COMPARISON_REPORT.md"):
        path = ROOT / rel
        if path.is_file():
            st.download_button(f"Download {rel}", path.read_bytes(), rel, mime="text/markdown")
    st.write("For an offline viva, install the pinned dependencies in advance, keep the saved `data/yfinance_rebuild` and `results/yfinance_rebuild` folders locally, and launch from the repository root. No page automatically downloads prices or refits a model.")
    st.markdown("[Project source on GitHub](https://github.com/ishan-s0323/ML-Stock-PORTFOLIO-MANAGEMENT)")


def main() -> None:
    st.sidebar.markdown("## ML Stock Portfolio Management")
    st.sidebar.caption("Saved-results research dashboard")
    guided = st.sidebar.toggle("Guided viva mode · 8–10 min", value=False)
    demo_condition = None
    if guided:
        step = st.sidebar.selectbox("Viva step", DEMO_LABELS)
        position = DEMO_LABELS.index(step)
        page = DEMO[position]
        if position == 5: demo_condition = "best"
        st.sidebar.progress((position + 1) / len(DEMO_LABELS))
        st.sidebar.caption("Follow the numbered steps; each page reads saved results.")
    else:
        page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.divider()
    st.sidebar.caption("Exploratory · Historical Simulation")
    st.sidebar.caption("Academic research only · no investment recommendations")
    functions = {
        "Research Overview": overview, "Stock Data Explorer": stock_explorer,
        "ML Model Performance": model_performance, "Shared-Capital Backtesting": portfolio_page,
        "Transaction Costs": costs_page, "Best vs Worst Windows": lambda: windows_page(demo_condition),
        "Bullish vs Bearish Regimes": regimes_page, "Exposure-Matched Analysis": matched_page,
        "ML Interpretability": interpretability_page, "Statistical Validation": stats_page,
        "Methodology & Limitations": methodology_page,
    }
    try:
        functions[page]()
    except DataUnavailable as exc:
        st.error(str(exc))
        st.info("This dashboard reads only the documented rebuilt yfinance outputs. Restore the required saved file or run the project's documented acquisition and experiments separately; nothing is downloaded or retrained automatically.")


if __name__ == "__main__":
    main()


