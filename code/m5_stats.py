# -*- coding: utf-8 -*-
"""Extract real M5 marginal statistics for simulator face-validity checks.

These statistics calibrate only observable first-order margins: demand
distribution, price distribution, price-change frequency, promotion lift, and a
reference WMAPE. They do not identify structural parameters such as hidden
confounding strength or exclusion-violation channels.
"""
import sys

import numpy as np

from forecasting_utils import build_panel


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def real_m5_stats(store="CA_3", cat="FOODS"):
    """Return (stats, weekly_panel) for real M5 calibration targets."""
    wk = build_panel(store, cat).copy()
    wk = wk.sort_values(["item_id", "wm_yr_wk"])

    wk["price_prev"] = wk.groupby("item_id")["sell_price"].shift(1)
    changed = (
        (wk["sell_price"].round(2) != wk["price_prev"].round(2))
        & wk["price_prev"].notna()
    )
    price_change_freq = float(changed.mean())

    wk["med_price"] = wk.groupby("item_id")["sell_price"].transform("median")
    on_promo = wk["sell_price"] < wk["med_price"] * 0.98
    lift = float(
        wk.loc[on_promo, "units"].mean()
        / max(wk.loc[~on_promo, "units"].mean(), 1e-9)
    )

    stats = {
        "n_sku": int(wk.item_id.nunique()),
        "n_week": int(wk.wm_yr_wk.nunique()),
        "n_obs": int(len(wk)),
        "units_mean": float(wk.units.mean()),
        "units_std": float(wk.units.std()),
        "units_cv": float(wk.units.std() / wk.units.mean()),
        "units_median": float(wk.units.median()),
        "logunits_mean": float(wk.log_units.mean()),
        "logunits_std": float(wk.log_units.std()),
        "logprice_mean": float(wk.log_price.mean()),
        "logprice_std": float(wk.log_price.std()),
        "price_change_freq": price_change_freq,
        "promo_lift_ratio": lift,
        "wmape_ref_gbm": 0.356,
    }
    return stats, wk


if __name__ == "__main__":
    print("Loading real M5 panel (CA_3/FOODS) and extracting calibration targets...")
    s, wk = real_m5_stats()
    print("\nReal M5 marginal statistics (face-validity targets):")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:10.4f}")
        else:
            print(f"  {k:20s}: {v:10d}")
