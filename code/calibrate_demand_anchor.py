from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as experiment


TARGET = np.array([23.57, 10.00, 2.41])


def simulate(seed: int, mean: float, sd: float):
    rng = np.random.default_rng(seed)
    n_sku, n_week = 500, 130
    week = np.arange(n_week)
    season = np.sin(2 * np.pi * week / 52) + 0.30 * experiment.unit_variance_laplace(
        rng, n_week
    )
    price_anchor = rng.normal(1.00, 0.57, n_sku)
    demand_anchor = rng.normal(mean, sd, n_sku)
    values = []
    for t in week:
        hidden = experiment.unit_variance_laplace(rng, n_sku)
        cost = experiment.unit_variance_laplace(rng, n_sku)
        price_noise = experiment.unit_variance_laplace(rng, n_sku)
        demand_noise = experiment.unit_variance_laplace(rng, n_sku)
        log_price = price_anchor + 0.15 * season[t] + 0.075 * cost + 0.15 * price_noise
        log_demand = (
            demand_anchor
            - 0.67 * (log_price - price_anchor)
            + 0.25 * season[t]
            + 0.35 * hidden
            + 0.30 * demand_noise
        )
        values.append(np.exp(log_demand))
    demand = np.concatenate(values)
    return np.array([demand.mean(), np.median(demand), demand.std() / demand.mean()])


def main():
    candidates = []
    for mean in [2.20, 2.25, 2.30, 2.35]:
        for sd in [1.12, 1.18, 1.24, 1.30]:
            metrics = np.mean([simulate(seed, mean, sd) for seed in range(10)], axis=0)
            relative_error = np.mean(np.abs(metrics / TARGET - 1))
            candidates.append((relative_error, mean, sd, metrics))
    for result in sorted(candidates)[:8]:
        print(result)


if __name__ == "__main__":
    main()
