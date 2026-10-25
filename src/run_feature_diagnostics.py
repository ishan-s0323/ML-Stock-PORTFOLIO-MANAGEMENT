"""Exploratory 2021 feature importance, group ablation, and calibration.

Uses the rebuilt stock panel, fits only pre-2021 observations, and evaluates
2021. It does not alter trading-model, threshold, or portfolio outputs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from run_50_stock_benchmark import FEATURES, ROOT, XGB_PARAMS, load_stock, source_files

OUTPUT = ROOT / "results/yfinance_rebuild/feature_diagnostics"
GROUPS = {
    "RSI": ("RSI_14",),
    "MACD": ("MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"),
    "Bollinger": ("BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0"),
    "Trend": ("SMA_10", "SMA_20", "SMA_50", "SMA_200", "Close_vs_SMA20", "Close_vs_SMA50"),
    "Return/volume": ("Return_1d", "Volume_Change"),
}
MODELS = (("XGBoost 1D", 1), ("XGBoost 3D", 3), ("XGBoost 5D", 5),
          ("Logistic Regression 1D", 1))
START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2021-12-16")


def fit_predict(model_name, columns, train, valid, target):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train.loc[:, columns])
    x_valid = scaler.transform(valid.loc[:, columns])
    y = train[target].to_numpy(dtype=int)
    if model_name.startswith("XGBoost"):
        estimator = XGBClassifier(**XGB_PARAMS)
    else:
        estimator = LogisticRegression(C=.1, max_iter=500, random_state=42)
    estimator.fit(x_train, y)
    probability = estimator.predict_proba(x_valid)[:, 1]
    if model_name.startswith("XGBoost"):
        weights = estimator.feature_importances_.astype(float)
    else:
        weights = np.abs(estimator.coef_[0]).astype(float)
    weights = weights / weights.sum() if weights.sum() > 0 else weights
    return probability, dict(zip(columns, weights))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    score_rows, importance_rows, ablation_rows, probability_rows = [], [], [], []
    for number, path in enumerate(source_files(), 1):
        stock = path.name.removesuffix("_features.csv")
        df, _ = load_stock(path)
        endpoint = pd.Series(df.index, index=df.index).shift(-1)
        valid = df.loc[(df.index >= START) & (df.index < END) &
                       endpoint.notna() & df.loc[:, list(FEATURES)].notna().all(axis=1)].copy()
        for model_name, horizon in MODELS:
            target = f"target_{horizon}d"
            outcome_end = pd.Series(df.index, index=df.index).shift(-(horizon+1))
            train = df.loc[(df.index < START) & (outcome_end < START) &
                           df.loc[:, list(FEATURES)].notna().all(axis=1) &
                           df[target].notna()].copy()
            scored = valid.loc[valid[target].notna()].copy()
            if len(train) < 252 or len(scored) < 150:
                raise ValueError(f"Insufficient training/validation observations: {stock}, {model_name}")
            y = scored[target].to_numpy(dtype=int)
            if len(np.unique(y)) != 2:
                raise ValueError(f"One-class validation: {stock}, {model_name}")
            full_prob, weights = fit_predict(model_name, list(FEATURES), train, scored, target)
            full_auc = roc_auc_score(y, full_prob)
            full_brier = brier_score_loss(y, full_prob)
            score_rows.append({"Stock": stock, "Model": model_name, "Rows": len(scored),
                               "AUC": full_auc, "Brier": full_brier,
                               "LogLoss": log_loss(y, full_prob)})
            for feature, weight in weights.items():
                importance_rows.append({"Stock": stock, "Model": model_name,
                                        "Feature": feature, "Normalized_Importance": weight})
            for group, dropped in GROUPS.items():
                columns = [feature for feature in FEATURES if feature not in dropped]
                prob, _ = fit_predict(model_name, columns, train, scored, target)
                ablation_rows.append({
                    "Stock": stock, "Model": model_name, "Removed_Group": group,
                    "Full_AUC": full_auc, "Ablated_AUC": roc_auc_score(y, prob),
                    "AUC_Drop": full_auc - roc_auc_score(y, prob),
                    "Full_Brier": full_brier, "Ablated_Brier": brier_score_loss(y, prob),
                    "Brier_Increase": brier_score_loss(y, prob) - full_brier,
                })
            probability_rows.extend({"Stock": stock, "Model": model_name,
                                     "Date": date.date().isoformat(),
                                     "Probability": float(prob), "Outcome": int(outcome)}
                                    for date, prob, outcome in zip(scored.index, full_prob, y))
        print(f"Diagnosed {number}/50: {stock}", flush=True)
    score = pd.DataFrame(score_rows)
    importance = pd.DataFrame(importance_rows)
    ablation = pd.DataFrame(ablation_rows)
    probability = pd.DataFrame(probability_rows)
    score.to_csv(OUTPUT / "validation_scores.csv", index=False)
    importance.to_csv(OUTPUT / "feature_importance_stock.csv", index=False)
    ablation.to_csv(OUTPUT / "group_ablation_stock.csv", index=False)
    probability.to_csv(OUTPUT / "validation_probabilities.csv", index=False)
    importance.groupby(["Model", "Feature"], as_index=False).Normalized_Importance.mean().to_csv(
        OUTPUT / "feature_importance_summary.csv", index=False)
    ablation.groupby(["Model", "Removed_Group"], as_index=False)[["AUC_Drop", "Brier_Increase"]].mean().to_csv(
        OUTPUT / "group_ablation_summary.csv", index=False)
    calibration_rows = []
    for model, frame in probability.groupby("Model"):
        p = frame.Probability.to_numpy()
        y = frame.Outcome.to_numpy()
        bins = np.minimum(np.floor(p * 10).astype(int), 9)
        ece = 0.
        for b in range(10):
            mask = bins == b
            if not mask.any():
                continue
            confidence, frequency = float(p[mask].mean()), float(y[mask].mean())
            ece += mask.mean() * abs(confidence-frequency)
            calibration_rows.append({"Model": model, "Bin": b, "Count": int(mask.sum()),
                                     "Mean_Probability": confidence,
                                     "Observed_Frequency": frequency})
        score.loc[score.Model == model, "Pooled_ECE"] = ece
    pd.DataFrame(calibration_rows).to_csv(OUTPUT / "calibration_bins.csv", index=False)
    score.to_csv(OUTPUT / "validation_scores.csv", index=False)
    pd.DataFrame([{"Model": model, "Stocks": frame.Stock.nunique(), "Observations": len(frame),
                   "Pooled_AUC": roc_auc_score(frame.Outcome, frame.Probability),
                   "Pooled_Brier": brier_score_loss(frame.Outcome, frame.Probability),
                   "Pooled_LogLoss": log_loss(frame.Outcome, frame.Probability),
                   "Pooled_ECE": score.loc[score.Model == model, "Pooled_ECE"].iloc[0]}
                  for model, frame in probability.groupby("Model")]).to_csv(
        OUTPUT / "calibration_summary.csv", index=False)
    (OUTPUT / "diagnostic_manifest.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "training_rule": "train before 2021-01-01 with matured labels",
        "validation_range": "[2021-01-01, 2021-12-16)",
        "status": "exploratory; 2021 previously used for threshold selection",
        "importance": "mean per-stock normalized XGBoost gain or absolute standardized logistic coefficient",
        "ablation": "drop one prespecified disjoint feature group, refit pre-2021, score 2021",
        "calibration": "equal-width 0.1 bins pooled across stocks; ECE is descriptive",
        "features": list(FEATURES), "groups": GROUPS,
        "xgboost_parameters": XGB_PARAMS, "logistic_C": .1,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
