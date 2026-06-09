"""
Shared paired forecast-comparison tests for panel walk-forward experiments.

The experiment outputs are county-month panels, but Diebold-Mariano is a
time-series forecast-origin test.  To avoid treating counties from the same
month as independent forecast origins, these helpers first aggregate paired
losses within each forecast month and then test the monthly loss differential.

Sign convention:
    loss_diff = loss(model_a) - loss(model_b)

Positive statistics mean model_b has lower loss than model_a.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _round_or_none(value: float, ndigits: int = 6) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), ndigits)


def diebold_mariano_from_loss_diff(
    loss_diff: np.ndarray,
    *,
    horizon: int = 1,
    hac_lags: int = 1,
) -> dict[str, Any]:
    """
    Harvey-Leybourne-Newbold corrected Diebold-Mariano test.

    Args:
        loss_diff: Time-ordered loss differential. Positive means the second
            model in the comparison is more accurate.
        horizon: Forecast horizon. These experiments are one-step/month-ahead.
        hac_lags: Newey-West/Bartlett lags for serial correlation in monthly
            loss differentials.  Using 1 is conservative for adjacent monthly
            walk-forward origins.
    """
    d = np.asarray(loss_diff, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return {
            "dm_stat": None,
            "p_value": None,
            "significant_at_05": False,
            "n_forecast_origins": n,
            "mean_loss_diff": _round_or_none(float(np.mean(d)) if n else math.nan, 12),
            "hac_lags": hac_lags,
            "horizon": horizon,
            "note": "Too few forecast origins for Diebold-Mariano test.",
        }

    centered = d - d.mean()
    # Long-run variance of the monthly loss differential.
    lrv = float(np.dot(centered, centered) / n)
    max_lag = min(hac_lags, n - 1)
    for lag in range(1, max_lag + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        lrv += 2.0 * weight * gamma

    if lrv <= 0 or not np.isfinite(lrv):
        return {
            "dm_stat": None,
            "p_value": None,
            "significant_at_05": False,
            "n_forecast_origins": n,
            "mean_loss_diff": _round_or_none(float(d.mean()), 12),
            "hac_lags": hac_lags,
            "horizon": horizon,
            "note": "Non-positive long-run variance; test statistic undefined.",
        }

    dm = float(d.mean() / math.sqrt(lrv / n))
    h = horizon
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_adj = dm * hln
    p_value = float(2 * (1 - stats.t.cdf(abs(dm_adj), df=n - 1)))

    return {
        "dm_stat": round(dm_adj, 4),
        "p_value": round(p_value, 6),
        "significant_at_05": bool(p_value < 0.05),
        "n_forecast_origins": n,
        "mean_loss_diff": _round_or_none(float(d.mean()), 12),
        "hac_lags": max_lag,
        "horizon": horizon,
        "sign_convention": "positive means model_b has lower loss than model_a",
    }


def paired_panel_tests(
    df: pd.DataFrame,
    *,
    actual_col: str,
    model_a_pred_col: str,
    model_b_pred_col: str,
    date_col: str = "date",
    model_a_label: str = "model_a",
    model_b_label: str = "model_b",
    hac_lags: int = 1,
) -> dict[str, Any]:
    """
    Run panel-aware paired tests for two walk-forward prediction columns.

    Returns Diebold-Mariano on monthly mean squared-error differences and
    Wilcoxon signed-rank tests on monthly mean absolute-error differences.
    """
    needed = [date_col, actual_col, model_a_pred_col, model_b_pred_col]
    panel = df[needed].dropna().copy()
    panel[date_col] = pd.to_datetime(panel[date_col])
    panel["se_a"] = (panel[actual_col] - panel[model_a_pred_col]) ** 2
    panel["se_b"] = (panel[actual_col] - panel[model_b_pred_col]) ** 2
    panel["ae_a"] = (panel[actual_col] - panel[model_a_pred_col]).abs()
    panel["ae_b"] = (panel[actual_col] - panel[model_b_pred_col]).abs()

    monthly = (
        panel.groupby(date_col, sort=True)
        .agg(
            se_a=("se_a", "mean"),
            se_b=("se_b", "mean"),
            ae_a=("ae_a", "mean"),
            ae_b=("ae_b", "mean"),
            n=("se_a", "size"),
        )
        .reset_index()
    )
    monthly["squared_loss_diff"] = monthly["se_a"] - monthly["se_b"]
    monthly["absolute_loss_diff"] = monthly["ae_a"] - monthly["ae_b"]

    dm = diebold_mariano_from_loss_diff(
        monthly["squared_loss_diff"].to_numpy(),
        horizon=1,
        hac_lags=hac_lags,
    )

    wilcoxon: dict[str, Any]
    try:
        w_two = stats.wilcoxon(
            monthly["ae_a"],
            monthly["ae_b"],
            alternative="two-sided",
            zero_method="wilcox",
        )
        w_greater = stats.wilcoxon(
            monthly["ae_a"],
            monthly["ae_b"],
            alternative="greater",
            zero_method="wilcox",
        )
        wilcoxon = {
            "stat": round(float(w_two.statistic), 4),
            "p_value_two_sided": round(float(w_two.pvalue), 6),
            "p_value_model_b_better": round(float(w_greater.pvalue), 6),
            "significant_at_05_two_sided": bool(w_two.pvalue < 0.05),
            "significant_at_05_model_b_better": bool(w_greater.pvalue < 0.05),
            "n_forecast_origins": int(len(monthly)),
            "sign_convention": "model_b_better tests monthly AE(model_a) > AE(model_b)",
        }
    except ValueError as exc:
        wilcoxon = {
            "stat": None,
            "p_value_two_sided": None,
            "p_value_model_b_better": None,
            "significant_at_05_two_sided": False,
            "significant_at_05_model_b_better": False,
            "n_forecast_origins": int(len(monthly)),
            "note": str(exc),
        }

    return {
        "model_a": model_a_label,
        "model_b": model_b_label,
        "unit_of_analysis": "forecast month; county losses averaged within month",
        "n_panel_rows": int(len(panel)),
        "n_forecast_origins": int(len(monthly)),
        "mean_monthly_squared_loss_diff": _round_or_none(
            float(monthly["squared_loss_diff"].mean()), 12
        ),
        "mean_monthly_absolute_loss_diff": _round_or_none(
            float(monthly["absolute_loss_diff"].mean())
        ),
        "diebold_mariano": dm,
        "wilcoxon": wilcoxon,
    }
