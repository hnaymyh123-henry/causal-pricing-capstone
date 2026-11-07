from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as experiment


def first_stage_f(panel):
    train, _ = experiment.split_and_add_item_mean(panel)
    data = experiment.within_item(
        train, ["log_price", "cost_shock", "season"]
    )
    controls = data[["season"]].to_numpy()
    price = experiment.residualize(data["log_price"].to_numpy(), controls)
    instrument = experiment.residualize(data["cost_shock"].to_numpy(), controls)
    return experiment.first_stage_f_statistic(price, instrument)


def oracle_shares(world_name: str, seed: int = 0):
    world = experiment.experimental_worlds()[world_name]
    _, test = experiment.split_and_add_item_mean(
        experiment.generate_panel(world, seed)
    )
    candidate_prices = (
        test["price"].to_numpy()[:, None]
        * experiment.PRICE_MULTIPLIERS[None, :]
    )
    unit_cost = test["unit_cost"].to_numpy()[:, None]
    demand = experiment.simulator_demand_at_candidate_prices(
        test, world, np.log(experiment.PRICE_MULTIPLIERS)[None, :]
    )
    chosen = np.argmax(experiment.profit(candidate_prices, demand, unit_cost), axis=1)
    return np.bincount(chosen, minlength=len(experiment.PRICE_MULTIPLIERS)) / len(
        chosen
    )


def main():
    worlds = experiment.experimental_worlds()
    price_mean = []
    price_sd = []
    demand_mean = []
    demand_median = []
    demand_cv = []
    for seed in range(30):
        panel = experiment.generate_panel(worlds["baseline"], seed)
        demand = panel["simulator_demand"].to_numpy()
        price_mean.append(panel["log_price"].mean())
        price_sd.append(panel["log_price"].std())
        demand_mean.append(demand.mean())
        demand_median.append(np.median(demand))
        demand_cv.append(demand.std() / demand.mean())
    print(
        "30-seed baseline marginal means",
        {
            "log_price_mean": np.mean(price_mean),
            "log_price_sd": np.mean(price_sd),
            "demand_mean": np.mean(demand_mean),
            "demand_median": np.mean(demand_median),
            "demand_cv": np.mean(demand_cv),
        },
    )
    print(
        "30-seed between-seed SD",
        {
            "log_price_mean": np.std(price_mean, ddof=1),
            "log_price_sd": np.std(price_sd, ddof=1),
            "demand_mean": np.std(demand_mean, ddof=1),
            "demand_median": np.std(demand_median, ddof=1),
            "demand_cv": np.std(demand_cv, ddof=1),
        },
    )

    weak_f = [
        first_stage_f(experiment.generate_panel(worlds["weak_instrument"], seed))
        for seed in range(30)
    ]
    print(
        "weak conventional F 30 seeds",
        {
            "mean": np.mean(weak_f),
            "median": np.median(weak_f),
            "q10": np.quantile(weak_f, 0.10),
            "q90": np.quantile(weak_f, 0.90),
        },
    )

    for name in worlds:
        print(name, "oracle shares", np.round(oracle_shares(name), 4))

    _, test = experiment.split_and_add_item_mean(
        experiment.generate_panel(worlds["higher_elasticity"], 0)
    )
    ratio = test["unit_cost"].to_numpy() / test["price"].to_numpy()
    print("elastic reference unit_cost/current_price q05 q50 q95", np.quantile(ratio, [.05, .5, .95]))


if __name__ == "__main__":
    main()
