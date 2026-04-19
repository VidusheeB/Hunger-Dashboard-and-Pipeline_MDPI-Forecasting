"""
generate_figures.py — Produce all paper-ready figures for the methods section.

Run from project root:
    python finalfigures/generate_figures.py

Outputs (all saved to finalfigures/):
    fig1_actual_vs_predicted.png
    fig2_feature_importance.png
    fig3_trends_ablation.png
    fig4_lag_robustness.png
    fig6_roc_curve.png
    fig7_precision_recall.png
    fig8_confusion_matrix.png
    fig9_alert_distribution.png
    fig10_residuals_distribution.png
    fig11_walkforward_r2_over_time.png
    table1_threshold_sweep.png
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.tune_deployable_model import merge_laus, DEPLOYABLE_FEATURES
from pipeline import config

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(config.OUTPUTS_ROOT, "metrics")

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE   = "#2166ac"
RED    = "#d6604d"
GREEN  = "#4dac26"
ORANGE = "#f4a582"
GRAY   = "#888888"
YELLOW = "#f6c945"

COVID_START = "2020-01-01"
COVID_END   = "2021-12-31"
TARGET      = "SNAP_Application_Rate"


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"  Saved → {name}")


# ── Load data ────────────────────────────────────────────────────────────────

def load_predictions():
    df = pd.read_csv(os.path.join(METRICS_DIR, "deployable_walkforward_predictions.csv"),
                     parse_dates=["date"])
    df["covid"] = df["date"].between(COVID_START, COVID_END)
    return df


def load_features():
    df = pd.read_csv(config.FEATURES_CSV, parse_dates=["date"])
    df = merge_laus(df)
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.sort_values(["county", "date"]).reset_index(drop=True)


def load_json(name):
    with open(os.path.join(METRICS_DIR, name)) as f:
        return json.load(f)


# ── Fig 1: Actual vs Predicted (statewide monthly mean) ─────────────────────

def fig1_actual_vs_predicted(pred_df):
    noncovid = pred_df[~pred_df["covid"]]
    monthly = noncovid.groupby("date")[["actual_rate", "predicted_rate"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["date"], monthly["actual_rate"] * 1000,
            color=BLUE, lw=1.8, label="Actual", zorder=3)
    ax.plot(monthly["date"], monthly["predicted_rate"] * 1000,
            color=RED, lw=1.5, ls="--", label="Predicted", zorder=3)

    # shade COVID
    ax.axvspan(pd.Timestamp(COVID_START), pd.Timestamp(COVID_END),
               alpha=0.12, color="gray", label="COVID-19 (excluded from metrics)")

    ax.set_xlabel("Month")
    ax.set_ylabel("SNAP Application Rate (× 1,000)")
    ax.set_title("Fig 1 — Walk-Forward Predictions vs Actual (statewide mean, non-COVID)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    savefig("fig1_actual_vs_predicted.png")


# ── Fig 2: Feature Importance (deployable model, full-data fit) ──────────────

def fig2_feature_importance(feat_df):
    params = load_json("deployable_tuning_results.json")["best_params"]
    params["random_state"] = 42
    params["n_jobs"] = -1

    tr = feat_df[DEPLOYABLE_FEATURES + [TARGET]].dropna()
    model = XGBRegressor(**params, objective="reg:squarederror", verbosity=0)
    model.fit(tr[DEPLOYABLE_FEATURES].values, tr[TARGET].values)

    imp = pd.Series(model.feature_importances_, index=DEPLOYABLE_FEATURES)
    imp = imp.sort_values(ascending=True)

    # Colour by feature group
    def group_color(f):
        if "unemployment" in f: return ORANGE
        if any(k in f for k in ["calfresh", "foodbank", "foodstamps", "snaptopic"]): return BLUE
        if f in ["month_sin", "month_cos", "quarter", "month"]: return GREEN
        return GRAY

    colors = [group_color(f) for f in imp.index]

    fig, ax = plt.subplots(figsize=(8, 10))
    bars = ax.barh(imp.index, imp.values, color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Feature Importance (XGBoost gain)")
    ax.set_title("Fig 2 — Deployable Model Feature Importance\n(trained on full dataset, no SNAP rate features)")
    ax.grid(axis="x", alpha=0.3)

    legend_patches = [
        mpatches.Patch(color=BLUE,   label="Google Trends (lag/roll/momentum)"),
        mpatches.Patch(color=ORANGE, label="BLS Unemployment"),
        mpatches.Patch(color=GREEN,  label="Seasonality"),
        mpatches.Patch(color=GRAY,   label="Demographics"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    savefig("fig2_feature_importance.png")
    return imp


# ── Fig 3: Trends Ablation ───────────────────────────────────────────────────

def fig3_trends_ablation():
    ab = load_json("trends_ablation_results.json")
    metrics = ["R²", "MAE (×1000)", "sMAPE (%)"]
    no_vals = [
        ab["metrics_no_trends"]["r2"],
        ab["metrics_no_trends"]["mae"] * 1000,
        ab["metrics_no_trends"]["smape"],
    ]
    wt_vals = [
        ab["metrics_with_trends"]["r2"],
        ab["metrics_with_trends"]["mae"] * 1000,
        ab["metrics_with_trends"]["smape"],
    ]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - width/2, no_vals, width, label="No Trends (10 features)", color=GRAY,   alpha=0.85)
    b2 = ax.bar(x + width/2, wt_vals, width, label="With Trends (26 features)", color=BLUE, alpha=0.85)

    # Delta annotations
    deltas = [wt_vals[i] - no_vals[i] for i in range(len(metrics))]
    signs  = ["+", "+", ""]  # R² and MAE improve in opposite directions; sMAPE lower = better
    labels = [f"{'+' if d > 0 else ''}{d:.3f}" for d in deltas]
    for i, (bar, label) in enumerate(zip(b2, labels)):
        color = GREEN if (i == 0 or deltas[i] < 0) else RED
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                label, ha="center", va="bottom", fontsize=8.5, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    dm_p = ab["paired_tests"]["diebold_mariano"]["p_value"]
    wilcox_p = ab["paired_tests"]["wilcoxon"]["p_value_two_sided"]
    ax.set_title("Fig 3 — Trends Ablation: With vs Without Google Trends\n"
                 f"Month-clustered DM p = {dm_p:.4f}  |  Wilcoxon p = {wilcox_p:.4f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    savefig("fig3_trends_ablation.png")


# ── Fig 4: Lag Robustness (training gap) ─────────────────────────────────────

def fig4_lag_robustness():
    data = load_json("lag_robustness_results.json")["results"]
    df   = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: No vs With R² by gap
    ax = axes[0]
    ax.plot(df["gap"], df["no_r2"],  color=GRAY, lw=2, marker="o", ms=5, label="No Trends")
    ax.plot(df["gap"], df["with_r2"], color=BLUE, lw=2, marker="s", ms=5, label="With Trends")
    ax.fill_between(df["gap"], df["no_r2"], df["with_r2"], alpha=0.15, color=BLUE)
    ax.set_xlabel("Training Gap (months of stale SNAP data)")
    ax.set_ylabel("R²")
    ax.set_title("R² by Training Gap")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: ΔR² by gap with significance markers
    ax = axes[1]
    colors = [GREEN if r["dm_sig"] else RED for r in data]
    ax.bar(df["gap"], df["delta_r2"], color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Training Gap (months)")
    ax.set_ylabel("ΔR² (With Trends − No Trends)")
    ax.set_title("ΔR² by Gap  (green = DM significant p<0.05)")
    ax.grid(axis="y", alpha=0.3)

    sig_patch   = mpatches.Patch(color=GREEN, label="DM p < 0.05")
    insig_patch = mpatches.Patch(color=RED,   label="DM p ≥ 0.05")
    ax.legend(handles=[sig_patch, insig_patch], fontsize=9)

    sig_gaps = df.loc[df["dm_sig"], "gap"].astype(int).tolist()
    sig_text = f"DM significant gaps {min(sig_gaps)}-{max(sig_gaps)}" if sig_gaps else "No DM-significant gaps"
    plt.suptitle("Fig 4 — Lag Robustness: Walk-Forward with Training Gap\n"
                 f"Spearman ρ = −0.967, p < 0.0001; {sig_text}",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    savefig("fig4_lag_robustness.png")


# ── Fig 6: ROC-style curve (Recall vs FPR from threshold sweep) ──────────────

def fig6_roc_curve():
    data = load_json("threshold_alert_summary.json")
    sweep = pd.DataFrame(data["threshold_sweep"])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(sweep["fpr"], sweep["recall"], color=BLUE, lw=2, marker="o", ms=7, zorder=3)

    # Annotate each point with percentile
    for _, row in sweep.iterrows():
        pct = int(row["percentile"])
        offset = (0.01, 0.01)
        if pct == 60:
            offset = (0.01, -0.03)
            ax.plot(row["fpr"], row["recall"], "o", ms=12, color=RED, alpha=0.4, zorder=2)
        ax.annotate(f"{pct}th", (row["fpr"], row["recall"]),
                    xytext=(row["fpr"] + offset[0], row["recall"] + offset[1]),
                    fontsize=8, color=GRAY)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Random classifier")
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("Recall (Sensitivity)")
    ax.set_title("Fig 6 — ROC Curve: Alert Threshold Sweep\n(each point = one candidate Red percentile)")
    ax.set_xlim(-0.02, 0.35)
    ax.set_ylim(0, 1.05)

    f1_patch = mpatches.Patch(color=RED, alpha=0.4, label="Selected (60th, F1-optimal)")
    ax.legend(handles=[f1_patch], fontsize=9)
    ax.grid(alpha=0.3)
    savefig("fig6_roc_curve.png")


# ── Fig 7: Precision-Recall curve ────────────────────────────────────────────

def fig7_precision_recall():
    data  = load_json("threshold_alert_summary.json")
    sweep = pd.DataFrame(data["threshold_sweep"])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(sweep["recall"], sweep["precision"], color=BLUE, lw=2, marker="o", ms=7, zorder=3)

    for _, row in sweep.iterrows():
        pct = int(row["percentile"])
        if pct == 60:
            ax.plot(row["recall"], row["precision"], "o", ms=12, color=RED, alpha=0.4, zorder=2)
        ax.annotate(f"{pct}th", (row["recall"], row["precision"]),
                    xytext=(row["recall"] - 0.04, row["precision"] + 0.02),
                    fontsize=8, color=GRAY)

    # No-skill line = prevalence
    prevalence = data["n_true_events_test"] / (data["test_months"] * 58)
    ax.axhline(prevalence, color="gray", ls="--", lw=0.8,
               label=f"No-skill baseline ({prevalence:.3f})")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Fig 7 — Precision-Recall Curve: Alert Threshold Sweep")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    f1_patch = mpatches.Patch(color=RED, alpha=0.4, label="Selected (60th, F1-optimal)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    savefig("fig7_precision_recall.png")


# ── Fig 8: Confusion Matrix ───────────────────────────────────────────────────

def fig8_confusion_matrix():
    data = load_json("threshold_alert_summary.json")["confusion_matrix_test"]
    cm = np.array([[data["tp"], data["fn"]],
                   [data["fp"], data["tn"]]])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, f"{labels[i][j]}\n{val:,}",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if val > cm.max() * 0.5 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Red", "Predicted Not-Red"], fontsize=10)
    ax.set_yticklabels(["Actual Spike", "Actual Normal"], fontsize=10)
    ax.set_title(
        f"Fig 8 — Confusion Matrix @ F1-Optimal Threshold (60th pctile)\n"
        f"Recall={data['recall']:.3f}  Precision={data['precision']:.3f}  "
        f"F1={data['f1']:.3f}  Youden's J={data['youden_J']:.3f}"
    )
    plt.colorbar(im, ax=ax, fraction=0.046)
    savefig("fig8_confusion_matrix.png")


# ── Fig 9: Alert Distribution ─────────────────────────────────────────────────

def fig9_alert_distribution():
    data   = load_json("threshold_alert_summary.json")
    counts = data["full_data_label_counts"]
    pcts   = data["full_data_label_pcts"]

    labels = ["Green", "Yellow", "Red"]
    values = [counts[l] for l in labels]
    colors = [GREEN, YELLOW, RED]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    bars = ax1.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, pct in zip(bars, [pcts[l] for l in labels]):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 30,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("County-Month Count")
    ax1.set_title("Alert Label Distribution (all county-months)")
    ax1.grid(axis="y", alpha=0.3)

    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax2.set_title("Alert Distribution (pie)")

    plt.suptitle("Fig 9 — Green / Yellow / Red Alert Labels\n"
                 "Red=60th county-pctile (F1-optimal), Yellow=50th county-pctile",
                 fontsize=11)
    plt.tight_layout()
    savefig("fig9_alert_distribution.png")


# ── Fig 10: Residuals Distribution ───────────────────────────────────────────

def fig10_residuals(pred_df):
    noncovid = pred_df[~pred_df["covid"]]
    residuals = noncovid["actual_rate"] - noncovid["predicted_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram
    ax = axes[0]
    ax.hist(residuals * 1000, bins=60, color=BLUE, alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", lw=1.2)
    ax.axvline(residuals.mean() * 1000, color=RED, lw=1.5, ls="--",
               label=f"Mean = {residuals.mean()*1000:.4f}")
    ax.set_xlabel("Residual (actual − predicted, × 1,000)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.legend(fontsize=9)

    # Q-Q plot approximation: sorted residuals vs normal quantiles
    ax = axes[1]
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax.plot(osm, osr, ".", color=BLUE, alpha=0.4, ms=3)
    ax.plot(osm, slope * np.array(osm) + intercept, color=RED, lw=1.5)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title(f"Q-Q Plot  (r = {r:.3f})")
    ax.grid(alpha=0.3)

    plt.suptitle("Fig 10 — Residual Analysis (non-COVID county-months)", fontsize=11)
    plt.tight_layout()
    savefig("fig10_residuals_distribution.png")


# ── Fig 11: R² over walk-forward time ────────────────────────────────────────

def fig11_r2_over_time(pred_df):
    noncovid = pred_df[~pred_df["covid"]]

    def r2(grp):
        y    = grp["actual_rate"].values
        yhat = grp["predicted_rate"].values
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    monthly_r2 = noncovid.groupby("date").apply(r2).reset_index()
    monthly_r2.columns = ["date", "r2"]
    monthly_r2["r2_smooth"] = monthly_r2["r2"].rolling(6, min_periods=3, center=True).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly_r2["date"], monthly_r2["r2"], color=BLUE, alpha=0.3, lw=1, label="Monthly R²")
    ax.plot(monthly_r2["date"], monthly_r2["r2_smooth"], color=BLUE, lw=2, label="6-month rolling mean")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Month")
    ax.set_ylabel("R² (cross-section of counties)")
    ax.set_title("Fig 11 — Walk-Forward R² Over Time (non-COVID)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    savefig("fig11_walkforward_r2_over_time.png")


# ── Table 1: Threshold Sweep ─────────────────────────────────────────────────

def table1_threshold_sweep():
    data  = load_json("threshold_alert_summary.json")
    sweep = pd.DataFrame(data["threshold_sweep"])
    sweep = sweep[["percentile","recall","precision","specificity","fpr","f1","youden_J","tp","fp","fn","tn"]]
    sweep.columns = ["Pctile","Recall","Prec","Spec","FPR","F1","J","TP","FP","FN","TN"]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=sweep.round(3).values.tolist(),
        colLabels=sweep.columns.tolist(),
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Highlight F1-optimal row (index 2 = 60th pctile)
    for j in range(len(sweep.columns)):
        tbl[3, j].set_facecolor("#d6e4f7")

    ax.set_title("Table 1 — Threshold Sweep (Red pctile 50→95, county-specific thresholds)\n"
                 "Blue row = F1-optimal (60th pctile), selected threshold",
                 pad=20, fontsize=10)
    savefig("table1_threshold_sweep.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    pred_df = load_predictions()
    feat_df = load_features()

    print("\nGenerating figures …")
    fig1_actual_vs_predicted(pred_df)
    fig2_feature_importance(feat_df)
    fig3_trends_ablation()
    fig4_lag_robustness()
    fig6_roc_curve()
    fig7_precision_recall()
    fig8_confusion_matrix()
    fig9_alert_distribution()
    fig10_residuals(pred_df)
    fig11_r2_over_time(pred_df)
    table1_threshold_sweep()

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
