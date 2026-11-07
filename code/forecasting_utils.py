# -*- coding: utf-8 -*-
"""Shared utilities for CausalCast forecasting experiments.

The experiment scripts keep the final evaluation on real M5 held-out weeks.
Simulation is only used to train, augment, or select models.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "m5_diagnostics"
DATA_DIR = PROJECT_ROOT / "data" / "m5"

SEEDS = list(range(10))
REAL_FULL_FEATURES = [
    "log_price",
    "season",
    "snap",
    "nevent",
    "log_lag",
    "month",
    "year",
]
SHARED_FEATURES = ["log_price", "season", "log_lag", "event"]

DEFAULT_HGB_PARAMS = {
    "max_iter": 240,
    "learning_rate": 0.06,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.02,
    "early_stopping": False,
}


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR


def json_ready(obj):
    """Convert numpy/pandas objects into JSON-serializable Python objects."""
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, pd.Series):
        return json_ready(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return json_ready(obj.to_dict(orient="records"))
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, ensure_ascii=False)


def build_panel(store: str = "CA_3", cat: str = "FOODS") -> pd.DataFrame:
    """Build the real M5 weekly panel used by all retained experiments."""
    cal = pd.read_csv(DATA_DIR / "calendar.csv")
    sp = pd.read_csv(DATA_DIR / "sell_prices.csv")
    ss = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")

    cal["event_flag"] = cal["event_name_1"].notna().astype(int)
    snap_col = f"snap_{store[:2]}"
    calw = (
        cal.groupby("wm_yr_wk")
        .agg(
            snap=(snap_col, "sum"),
            nevent=("event_flag", "sum"),
            month=("month", "first"),
            year=("year", "first"),
        )
        .reset_index()
    )

    sub = ss[(ss.store_id == store) & (ss.cat_id == cat)].copy()
    dcols = [c for c in ss.columns if c.startswith("d_")]
    long = sub.melt(
        id_vars=["item_id"],
        value_vars=dcols,
        var_name="d",
        value_name="units",
    )
    long = long.merge(cal[["d", "wm_yr_wk"]], on="d")
    wk = long.groupby(["item_id", "wm_yr_wk"], as_index=False).agg(
        units=("units", "sum"),
        ndays=("units", "size"),
    )
    wk = wk.merge(
        sp[sp.store_id == store][["item_id", "wm_yr_wk", "sell_price"]],
        on=["item_id", "wm_yr_wk"],
        how="left",
    )
    wk = wk.dropna(subset=["sell_price"])
    wk = wk.merge(calw, on="wm_yr_wk")
    wk = wk.sort_values(["item_id", "wm_yr_wk"])
    wk["lag_units"] = wk.groupby("item_id")["units"].shift(1)
    wk = wk.dropna(subset=["lag_units"])
    wk["log_units"] = np.log1p(wk["units"])
    wk["log_price"] = np.log(wk["sell_price"])
    wk["log_lag"] = np.log1p(wk["lag_units"])

    season_by_month = wk.groupby("month")["units"].mean()
    wk["season"] = wk["month"].map(
        (season_by_month - season_by_month.mean()) / season_by_month.std()
    )
    return wk


def load_real_panel(store: str = "CA_3", cat: str = "FOODS") -> pd.DataFrame:
    df = build_panel(store=store, cat=cat).copy()
    df["event"] = df["nevent"].astype(float)
    return df


def prepare_sim_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event"] = out["event"].astype(float)
    out["log_lag"] = np.log1p(out["lag_units"].clip(lower=0))
    return out


def walk_forward_split(
    df: pd.DataFrame,
    week_col: str,
    train_frac: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    weeks = np.array(sorted(df[week_col].dropna().unique()))
    cut_idx = int(len(weeks) * train_frac)
    if cut_idx <= 0 or cut_idx >= len(weeks):
        raise ValueError(f"Bad walk-forward split for {len(weeks)} weeks.")

    train_weeks = set(weeks[:cut_idx])
    train = df[df[week_col].isin(train_weeks)].copy()
    test = df[~df[week_col].isin(train_weeks)].copy()
    meta = {
        "week_col": week_col,
        "n_week": int(len(weeks)),
        "train_frac": train_frac,
        "n_train_week": int(cut_idx),
        "n_test_week": int(len(weeks) - cut_idx),
        "train_start": int(weeks[0]),
        "train_end": int(weeks[cut_idx - 1]),
        "test_start": int(weeks[cut_idx]),
        "test_end": int(weeks[-1]),
    }
    return train, test, meta


def wmape(actual_units: Iterable[float], pred_units: Iterable[float]) -> float:
    actual = np.asarray(actual_units, dtype=float)
    pred = np.asarray(pred_units, dtype=float)
    return float(np.abs(actual - pred).sum() / max(actual.sum(), 1e-9))


def mase(
    actual_units: Iterable[float],
    pred_units: Iterable[float],
    scale: float,
) -> float:
    actual = np.asarray(actual_units, dtype=float)
    pred = np.asarray(pred_units, dtype=float)
    return float(np.abs(actual - pred).mean() / max(scale, 1e-9))


def naive_scale(train: pd.DataFrame, item_ids: Iterable | None = None) -> float:
    d = train
    if item_ids is not None:
        ids = set(item_ids)
        d = d[d["item_id"].isin(ids)]
        if d.empty:
            d = train
    err = (d["units"].astype(float) - d["lag_units"].astype(float)).abs()
    scale = float(err.replace([np.inf, -np.inf], np.nan).dropna().mean())
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(max(train["units"].mean(), 1.0))
    return scale


def slice_masks(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sparse_quantile: float = 0.20,
    min_sparse_nonzero_weeks: int = 52,
    cold_nonzero_weeks: int = 8,
    cold_total_weeks: int = 13,
) -> tuple[dict[str, np.ndarray], dict]:
    train_rows = train.groupby("item_id").size()
    train_nonzero = train[train["units"] > 0].groupby("item_id").size()
    row_count = test["item_id"].map(train_rows).fillna(0)
    nonzero_count = test["item_id"].map(train_nonzero).fillna(0)

    item_nonzero = (
        train_nonzero.reindex(test["item_id"].drop_duplicates()).fillna(0)
    )
    sparse_threshold = max(
        min_sparse_nonzero_weeks,
        float(item_nonzero.quantile(sparse_quantile)),
    )
    sparse = nonzero_count <= sparse_threshold
    cold = (
        (row_count == 0)
        | (nonzero_count <= cold_nonzero_weeks)
        | (row_count <= cold_total_weeks)
    )
    masks = {
        "overall": np.ones(len(test), dtype=bool),
        "sparse": sparse.to_numpy(dtype=bool),
        "cold_start": cold.to_numpy(dtype=bool),
    }
    meta = {
        "sparse_definition": (
            "test rows whose item has train nonzero weeks <= "
            f"max({min_sparse_nonzero_weeks}, q{int(sparse_quantile * 100)})"
        ),
        "sparse_nonzero_threshold": float(sparse_threshold),
        "cold_start_definition": (
            "train rows == 0, or train nonzero weeks <= "
            f"{cold_nonzero_weeks}, or train total weeks <= {cold_total_weeks}"
        ),
        "test_items": int(test["item_id"].nunique()),
        "sparse_items": int(test.loc[masks["sparse"], "item_id"].nunique()),
        "cold_start_items": int(
            test.loc[masks["cold_start"], "item_id"].nunique()
        ),
    }
    return masks, meta


def evaluate_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    pred_units: Iterable[float],
) -> tuple[dict, dict]:
    pred = np.asarray(pred_units, dtype=float).clip(0, None)
    masks, slice_meta = slice_masks(train, test)
    metrics = {}
    for name, mask in masks.items():
        d = test.loc[mask]
        if d.empty:
            metrics[name] = {
                "n_obs": 0,
                "n_item": 0,
                "actual_sum": 0.0,
                "wmape": None,
                "mase": None,
            }
            continue
        item_ids = d["item_id"].unique()
        scale = naive_scale(train, item_ids=item_ids)
        metrics[name] = {
            "n_obs": int(len(d)),
            "n_item": int(d["item_id"].nunique()),
            "actual_sum": float(d["units"].sum()),
            "wmape": wmape(d["units"].to_numpy(), pred[mask]),
            "mase": mase(d["units"].to_numpy(), pred[mask], scale),
            "naive_scale": scale,
        }
    return metrics, slice_meta


def fit_hgb(
    train: pd.DataFrame,
    features: list[str],
    seed: int,
    params: dict | None = None,
    sample_weight: np.ndarray | None = None,
) -> HistGradientBoostingRegressor:
    cfg = dict(DEFAULT_HGB_PARAMS)
    if params:
        cfg.update(params)
    cfg["random_state"] = seed
    model = HistGradientBoostingRegressor(**cfg)
    model.fit(
        train[features].to_numpy(dtype=float),
        train["log_units"].to_numpy(dtype=float),
        sample_weight=sample_weight,
    )
    return model


def predict_units(model: HistGradientBoostingRegressor, df: pd.DataFrame,
                  features: list[str]) -> np.ndarray:
    log_pred = model.predict(df[features].to_numpy(dtype=float))
    return np.expm1(log_pred).clip(0, None)


def mean_ci(values: Iterable[float]) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return {"mean": None, "ci95": None, "std": None, "n": 0}
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) > 1:
        try:
            from scipy import stats

            tval = float(stats.t.ppf(0.975, len(arr) - 1))
        except Exception:
            # Two-sided 95% critical values. Fall back to normal for larger n.
            tcrit = {
                1: 12.706,
                2: 4.303,
                3: 3.182,
                4: 2.776,
                5: 2.571,
                6: 2.447,
                7: 2.365,
                8: 2.306,
                9: 2.262,
                10: 2.228,
                15: 2.131,
                20: 2.086,
                30: 2.042,
            }
            df = len(arr) - 1
            tval = tcrit.get(df, 1.96 if df > 30 else 2.262)
        ci95 = tval * std / math.sqrt(len(arr))
    else:
        ci95 = 0.0
    return {
        "mean": float(arr.mean()),
        "ci95": float(ci95),
        "std": std,
        "n": int(len(arr)),
    }


def summarize_metric_records(records: list[dict]) -> dict:
    """Summarize rows with method/slice/wmape/mase columns."""
    summary: dict[str, dict] = {}
    for method in sorted({r["method"] for r in records}):
        summary[method] = {}
        method_rows = [r for r in records if r["method"] == method]
        for slice_name in sorted({r["slice"] for r in method_rows}):
            rows = [r for r in method_rows if r["slice"] == slice_name]
            summary[method][slice_name] = {
                "wmape": mean_ci([r["wmape"] for r in rows]),
                "mase": mean_ci([r["mase"] for r in rows]),
                "n_obs": int(rows[0]["n_obs"]) if rows else 0,
                "n_item": int(rows[0]["n_item"]) if rows else 0,
            }
    return summary


def flatten_metrics(seed: int, method: str, metrics: dict) -> list[dict]:
    rows = []
    for slice_name, vals in metrics.items():
        rows.append({
            "seed": seed,
            "method": method,
            "slice": slice_name,
            "wmape": vals["wmape"],
            "mase": vals["mase"],
            "n_obs": vals["n_obs"],
            "n_item": vals["n_item"],
            "actual_sum": vals["actual_sum"],
        })
    return rows


def print_summary_table(summary: dict, metric: str = "wmape") -> None:
    print(f"\n{metric.upper()} mean +/- 95% CI")
    print("-" * 78)
    print(f"{'method':<26}{'slice':<14}{'mean':>10}{'ci95':>10}{'n':>6}")
    print("-" * 78)
    for method, slices in summary.items():
        for slice_name, vals in slices.items():
            stat = vals[metric]
            mean = stat["mean"]
            ci95 = stat["ci95"]
            mean_s = "NA" if mean is None else f"{mean:.4f}"
            ci_s = "NA" if ci95 is None else f"{ci95:.4f}"
            print(f"{method:<26}{slice_name:<14}{mean_s:>10}{ci_s:>10}{stat['n']:>6}")
