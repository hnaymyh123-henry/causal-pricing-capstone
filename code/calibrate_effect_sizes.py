from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from m5_stats import real_m5_stats


def within_item(values, item_id):
    means = values.groupby(item_id).transform("mean")
    return values.to_numpy(dtype=float) - means.to_numpy(dtype=float)


def harmonic_summary(frame, target):
    y = within_item(frame[target], frame["item_id"])
    x = frame[["sin52", "cos52"]].to_numpy(dtype=float)
    model = LinearRegression().fit(x, y)
    amplitude = float(np.linalg.norm(model.coef_))
    residual = y - model.predict(x)
    return {
        "annual_log_amplitude": amplitude,
        "annual_peak_to_trough_ratio": float(np.exp(2.0 * amplitude)),
        "within_item_sd": float(np.std(y, ddof=1)),
        "post_harmonic_residual_sd": float(np.std(residual, ddof=1)),
    }


def main():
    stats, panel = real_m5_stats()
    weeks = np.sort(panel["wm_yr_wk"].unique())
    week_index = {week: index for index, week in enumerate(weeks)}
    t = panel["wm_yr_wk"].map(week_index).to_numpy(dtype=float)
    panel["sin52"] = np.sin(2.0 * np.pi * t / 52.0)
    panel["cos52"] = np.cos(2.0 * np.pi * t / 52.0)

    panel = panel.sort_values(["item_id", "wm_yr_wk"]).copy()
    log_price_change = panel.groupby("item_id")["log_price"].diff().abs()
    nonzero = log_price_change[log_price_change > 1e-12]

    demand_within = within_item(panel["log_units"], panel["item_id"])
    price_within = within_item(panel["log_price"], panel["item_id"])
    controls = np.column_stack(
        [price_within, panel["sin52"].to_numpy(), panel["cos52"].to_numpy()]
    )
    demand_residual = demand_within - LinearRegression().fit(
        controls, demand_within
    ).predict(controls)

    beta = -2.62
    cost_ratio = 0.65
    optimal_cost_price_ratio = 1.0 - 1.0 / abs(beta)
    optimal_multiplier_at_cost_ratio = cost_ratio / optimal_cost_price_ratio
    season_variance = 0.5 + 0.30**2
    clean_price_variance = 0.15**2 * season_variance + 0.075**2 + 0.15**2
    observed_control_ols_bias = (
        0.15 * 0.25 * season_variance / clean_price_variance
    )
    hidden_confounding_bias = 0.35**2 / (0.35**2 + 0.15**2)
    combined_unobserved_demand_sd = float(np.hypot(0.30, 0.35))

    payload = {
        "m5_scope": {
            "n_sku": int(stats["n_sku"]),
            "n_week": int(stats["n_week"]),
            "n_obs": int(stats["n_obs"]),
        },
        "m5_price": {
            **harmonic_summary(panel, "log_price"),
            "median_abs_nonzero_log_change": float(nonzero.median()),
            "median_abs_nonzero_pct_change": float(np.expm1(nonzero.median())),
            "q25_abs_nonzero_log_change": float(nonzero.quantile(0.25)),
            "q75_abs_nonzero_log_change": float(nonzero.quantile(0.75)),
        },
        "m5_demand": {
            **harmonic_summary(panel, "log_units"),
            "post_price_and_harmonic_residual_sd": float(
                np.std(demand_residual, ddof=1)
            ),
        },
        "analytical_cost_normalization": {
            "elasticity": beta,
            "chosen_cost_price_ratio": cost_ratio,
            "lerner_optimal_cost_price_ratio": optimal_cost_price_ratio,
            "implied_optimal_price_multiplier": optimal_multiplier_at_cost_ratio,
        },
        "simulator_effect_interpretation": {
            "price_log_0_15_one_sd_pct": float(np.expm1(0.15)),
            "price_season_peak_to_trough_ratio": float(np.exp(2.0 * 0.15)),
            "demand_season_peak_to_trough_ratio": float(np.exp(2.0 * 0.25)),
            "demand_noise_0_30_one_sd_pct": float(np.expm1(0.30)),
            "hidden_demand_0_35_one_sd_pct": float(np.expm1(0.35)),
            "combined_unobserved_demand_sd": combined_unobserved_demand_sd,
            "combined_sd_as_share_of_m5_residual_sd": float(
                combined_unobserved_demand_sd
                / np.std(demand_residual, ddof=1)
            ),
            "hidden_share_of_unobserved_variance": float(
                0.35**2 / (0.35**2 + 0.30**2)
            ),
            "season_noise_share_of_season_variance": float(
                0.30**2 / (0.5 + 0.30**2)
            ),
            "clean_world_expected_naive_ols_bias": observed_control_ols_bias,
            "hidden_world_expected_residual_bias": hidden_confounding_bias,
            "hidden_world_expected_beta_hat": beta + hidden_confounding_bias,
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
