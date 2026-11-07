"""Unified causal pricing experiment with three predictive decision baselines.

The script simulates retail data with a known structural price elasticity, hides that
answer from four causal methods and three predictive baselines, lets each choose prices,
and then scores the decisions with simulator-known demand. Comments are grouped
by section so the code remains readable while still explaining the full logic.

Run ``python experiment.py --quick`` for a short demonstration,
or run the file without arguments for all five worlds and thirty random seeds.
The DML branch uses week-grouped cross-fitting, and the DirectLiNGAM branch is
applied to within-item, non-Gaussian data so its baseline assumptions are clear.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from causallearn.search.FCMBased import lingam
from scipy.stats import t as student_t
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------------------------
# 1. Experimental worlds
# ---------------------------------------------------------------------------
#
# The experiment uses an inelastic boundary reference, an elastic decision
# reference, and three one-factor identification stresses applied to the elastic
# reference. beta is the structural price elasticity. hidden_price_effect
# controls confounding, instrument_effect controls how strongly the cost shock
# moves price, and direct_cost_effect deliberately breaks the IV exclusion rule.


@dataclass(frozen=True)
class World:
    label: str
    beta: float
    hidden_price_effect: float = 0.0
    instrument_effect: float = 0.075
    direct_cost_effect: float = 0.0


def experimental_worlds() -> dict[str, World]:
    return {
        "baseline": World("Inelastic\nreference", -0.67),
        "higher_elasticity": World("Elastic\nreference", -2.62),
        "weak_instrument": World("Weak\ninstrument", -2.62, instrument_effect=0.0018),
        "hidden_confounding": World(
            "Stronger hidden\nconfounding", -2.62, hidden_price_effect=0.35
        ),
        "exclusion_violated": World(
            "Exclusion\nviolated", -2.62, direct_cost_effect=0.075
        ),
    }


# ---------------------------------------------------------------------------
# 2. Synthetic data generation and chronological splitting
# ---------------------------------------------------------------------------
#
# generate_panel creates one row per SKU and week. Each product receives fixed
# baseline log-price and log-demand anchors. Hidden demand, the cost instrument,
# pricing noise, and demand noise are fresh independent Laplace shocks each week.
# The fixed anchors are Gaussian because within-item demeaning removes them
# before causal estimation. Non-Gaussian weekly shocks make the clean worlds
# compatible with DirectLiNGAM. The
# baseline does not let hidden demand move price; hidden_confounding adds that
# one channel as an explicit assumption-violation stress test. The returned
# panel deliberately omits hidden demand and structural elasticity.
# split_and_add_item_mean then uses the first 80% of weeks for training, the
# final 20% for testing, and creates a training-only product mean demand feature.
#
# Core relationships:
# log_price = price_anchor + 0.15*season + hidden_price_effect*hidden
#             + instrument_effect*cost_shock + 0.15*pricing_noise
# unit_cost = 0.65*exp(price_anchor + instrument_effect*cost_shock)
# price_effect = beta*(log_price - price_anchor)
# log_demand = demand_anchor + price_effect + 0.25*season + 0.35*hidden
#              + direct_cost_effect*cost_shock + 0.30*demand_noise
# demand = exp(log_demand), so beta is exactly the elasticity of unit demand.
# Positive terms raise the outcome and negative terms lower it; beta is negative, so a price increase normally reduces demand.


def unit_variance_laplace(rng: np.random.Generator, size: int) -> np.ndarray:
    """Draw mean-zero Laplace shocks scaled to unit variance."""
    return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size)


def generate_panel(
    world: World,
    seed: int,
    n_sku: int = 500,
    n_week: int = 130,
) -> pd.DataFrame:
    # The anchors are calibrated to the retained M5 CA_3/FOODS marginal targets.
    # The remaining coefficients have explicit design targets. Price noise 0.15
    # is close to M5's 0.123 median non-zero log-price change. Season loadings
    # 0.15/0.25 create about +0.53 omitted-season bias for Naive OLS. Demand
    # noise 0.30 and hidden demand 0.35 give a conservative combined SD of 0.461;
    # loading hidden demand into price at 0.35 targets about +0.845 confounding
    # bias. The 0.65 cost ratio places the beta=-2.62 interior optimum near the
    # +5% grid point by the constant-elasticity inverse-elasticity rule.
    log_price_mean = 1.00
    price_anchor_sd = 0.57
    base_demand_mean = 2.30
    base_demand_sd = 1.24
    demand_noise_sd = 0.30
    pricing_noise_sd = 0.15
    cost_baseline_ratio = 0.65
    season_effect = 0.25
    hidden_demand_effect = 0.35
    seasonal_price_effect = 0.15

    rng = np.random.default_rng(seed)

    week = np.arange(n_week)
    season = np.sin(2 * np.pi * week / 52) + 0.3 * unit_variance_laplace(rng, n_week)

    price_anchor = rng.normal(log_price_mean, price_anchor_sd, n_sku)
    demand_anchor = rng.normal(base_demand_mean, base_demand_sd, n_sku)

    rows: list[pd.DataFrame] = []

    for t in range(n_week):
        hidden_state = unit_variance_laplace(rng, n_sku)
        cost_shock = unit_variance_laplace(rng, n_sku)
        pricing_noise = unit_variance_laplace(rng, n_sku)

        log_price = (
            price_anchor
            + seasonal_price_effect * season[t]
            + world.hidden_price_effect * hidden_state
            + world.instrument_effect * cost_shock
            + pricing_noise_sd * pricing_noise
        )
        unit_cost = cost_baseline_ratio * np.exp(
            price_anchor + world.instrument_effect * cost_shock
        )

        price_delta = log_price - price_anchor
        price_effect = world.beta * price_delta

        log_demand = (
            demand_anchor
            + price_effect
            + season_effect * season[t]
            + hidden_demand_effect * hidden_state
            + world.direct_cost_effect * cost_shock
            + demand_noise_sd * unit_variance_laplace(rng, n_sku)
        )
        demand = np.exp(log_demand)

        rows.append(
            pd.DataFrame(
                {
                    "item_id": np.arange(n_sku),
                    "week": t,
                    "price": np.exp(log_price),
                    "log_price": log_price.copy(),
                    "log_units": log_demand.copy(),
                    "simulator_demand": demand,
                    "unit_cost": unit_cost,
                    "season": season[t],
                    "cost_shock": cost_shock.copy(),
                }
            )
        )

    panel = pd.concat(rows, ignore_index=True)
    return panel


def split_and_add_item_mean(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weeks = np.sort(panel["week"].unique())
    cut = int(len(weeks) * 0.80)
    train = panel[panel["week"].isin(weeks[:cut])].copy()
    test = panel[panel["week"].isin(weeks[cut:])].copy()

    item_mean = train.groupby("item_id")["log_units"].mean()
    for frame in (train, test):
        frame["item_log_mean"] = frame["item_id"].map(item_mean)
    return train, test


# ---------------------------------------------------------------------------
# 3. Four price-elasticity estimators and a correlation benchmark
# ---------------------------------------------------------------------------
#
# The main purpose of this section is to set up four alternative ways to
# estimate beta, the causal effect of price on demand. Before every causal
# estimator, within_item removes each SKU's fixed mean so identification uses
# changes within the same product. Naive OLS uses the direct price-demand
# relationship. DML removes observed season and cost-shock effects using five
# held-out, week-grouped folds. IV uses the cost shock as its instrument after
# controlling for season and reports its
# first-stage F statistic. DirectLiNGAM learns a causal graph from standardized,
# within-item variables and extracts the price-to-demand edge. The clean worlds
# use non-Gaussian independent shocks; hidden_confounding intentionally violates
# the no-latent-confounder assumption. Separately, HGB and two tabular
# Transformers supply predictive decision baselines.

# Shared method inputs appear here because this is where estimation begins.
PREDICTIVE_FEATURES = ["log_price", "season", "cost_shock", "item_log_mean"]
DML_CONTROLS = ["season", "cost_shock"]
IV_CONTROLS = ["season"]
CAUSAL_METHODS = ["IV", "DML", "Naive", "DirectLiNGAM"]
PREDICTIVE_METHODS = ["HGB", "TransformerSmall", "TransformerLarge"]
METHOD_NAMES = CAUSAL_METHODS + PREDICTIVE_METHODS


def within_item(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {col: df[col] - df.groupby("item_id")[col].transform("mean") for col in columns}
    )


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return y - LinearRegression().fit(x, y).predict(x)


def cross_fitted_panel_residuals(
    train: pd.DataFrame,
    target: str,
    controls: list[str],
    n_splits: int = 5,
) -> np.ndarray:
    """Create held-out residuals using fold-specific item means and controls."""
    residuals = np.empty(len(train), dtype=float)
    groups = train["week"].to_numpy()
    columns = [target, *controls]
    splitter = GroupKFold(n_splits=n_splits)
    for fit_index, held_out_index in splitter.split(train, groups=groups):
        fit = train.iloc[fit_index]
        held_out = train.iloc[held_out_index]
        item_means = fit.groupby("item_id")[columns].mean()

        fit_within = pd.DataFrame(
            {
                column: fit[column].to_numpy()
                - fit["item_id"].map(item_means[column]).to_numpy()
                for column in columns
            }
        )
        held_out_within = pd.DataFrame(
            {
                column: held_out[column].to_numpy()
                - held_out["item_id"].map(item_means[column]).to_numpy()
                for column in columns
            }
        )
        nuisance_model = LinearRegression().fit(
            fit_within[controls].to_numpy(), fit_within[target].to_numpy()
        )
        residuals[held_out_index] = held_out_within[
            target
        ].to_numpy() - nuisance_model.predict(
            held_out_within[controls].to_numpy()
        )
    return residuals


def first_stage_f_statistic(price: np.ndarray, instrument: np.ndarray) -> float:
    """Compute the single-instrument first-stage F statistic."""
    instrument_2d = instrument[:, None]
    fitted = LinearRegression().fit(instrument_2d, price).predict(instrument_2d)
    residual = price - fitted
    total_sum_squares = float(price @ price)
    residual_sum_squares = float(residual @ residual)
    r_squared = 1.0 - residual_sum_squares / max(total_sum_squares, 1e-15)
    return float(r_squared / max(1.0 - r_squared, 1e-15) * max(len(price) - 2, 1))


def estimate_betas(train: pd.DataFrame) -> tuple[dict[str, float], float]:
    basic = within_item(train, ["log_units", "log_price"])
    y = basic["log_units"].to_numpy()
    price = basic["log_price"].to_numpy()
    naive = float(np.polyfit(price, y, 1)[0])

    demand_residual = cross_fitted_panel_residuals(
        train, "log_units", DML_CONTROLS
    )
    price_residual = cross_fitted_panel_residuals(
        train, "log_price", DML_CONTROLS
    )
    dml = float((price_residual @ demand_residual) / (price_residual @ price_residual))

    iv_data = within_item(
        train, ["log_units", "log_price", "cost_shock", *IV_CONTROLS]
    )
    controls = iv_data[IV_CONTROLS].to_numpy()
    ry = residualize(iv_data["log_units"].to_numpy(), controls)
    rp = residualize(iv_data["log_price"].to_numpy(), controls)
    rz = residualize(iv_data["cost_shock"].to_numpy(), controls)
    iv = float((rz @ ry) / (rz @ rp))
    iv_first_stage_f = first_stage_f_statistic(rp, rz)

    columns = ["cost_shock", "season", "log_price", "log_units"]
    sample = within_item(train, columns)
    standardized = ((sample - sample.mean()) / sample.std().replace(0, 1)).to_numpy()
    model = lingam.DirectLiNGAM()
    model.fit(standardized)
    edge = float(
        model.adjacency_matrix_[columns.index("log_units"), columns.index("log_price")]
    )
    lingam_beta = edge * float(sample["log_units"].std() / sample["log_price"].std())

    return (
        {
            "Naive": naive,
            "DML": dml,
            "IV": iv,
            "DirectLiNGAM": lingam_beta,
        },
        iv_first_stage_f,
    )


class NumericalTransformer(nn.Module):
    """One token per numerical feature, plus a learned CLS token."""

    def __init__(self, n_features: int, width: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        self.feature_weight = nn.Parameter(torch.randn(n_features, width) * 0.02)
        self.feature_bias = nn.Parameter(torch.randn(n_features, width) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=2 * width, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.feature_weight + self.feature_bias
        cls = self.cls_token.expand(len(x), -1, -1)
        return self.head(self.encoder(torch.cat((cls, tokens), dim=1))[:, 0]).squeeze(-1)


class TransformerRegressor:
    """Training-set standardized wrapper with a scikit-learn-like predict API."""

    def __init__(self, seed: int, width: int = 32, n_layers: int = 2, epochs: int = 12) -> None:
        torch.manual_seed(seed)
        torch.set_num_threads(4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = NumericalTransformer(len(PREDICTIVE_FEATURES), width, n_layers).to(self.device)
        self.epochs = epochs

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TransformerRegressor":
        self.x_mean = x.mean(axis=0)
        self.x_std = np.maximum(x.std(axis=0), 1e-6)
        self.y_mean = float(y.mean())
        self.y_std = max(float(y.std()), 1e-6)
        x_tensor = torch.from_numpy(((x - self.x_mean) / self.x_std).astype(np.float32)).to(self.device)
        y_tensor = torch.from_numpy(((y - self.y_mean) / self.y_std).astype(np.float32)).to(self.device)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=1e-3, weight_decay=1e-4)
        batch_size = 2048
        for epoch in range(self.epochs):
            generator = torch.Generator().manual_seed(10000 + epoch)
            for indices in torch.randperm(len(x_tensor), generator=generator).split(batch_size):
                indices = indices.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.network(x_tensor[indices])
                loss = nn.functional.mse_loss(prediction, y_tensor[indices])
                loss.backward()
                optimizer.step()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.network.eval()
        x_tensor = torch.from_numpy(((x - self.x_mean) / self.x_std).astype(np.float32)).to(self.device)
        with torch.no_grad():
            outputs = [self.network(batch).cpu().numpy() for batch in x_tensor.split(4096)]
        return np.concatenate(outputs) * self.y_std + self.y_mean


def fit_hgb_model(train: pd.DataFrame, seed: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(early_stopping=False, random_state=seed)
    return model.fit(
        train[PREDICTIVE_FEATURES].to_numpy(dtype=float),
        train["log_units"].to_numpy(dtype=float),
    )


def fit_transformer_model(
    train: pd.DataFrame, seed: int, width: int, layers: int
) -> TransformerRegressor:
    return TransformerRegressor(seed, width=width, n_layers=layers).fit(
        train[PREDICTIVE_FEATURES].to_numpy(dtype=float),
        train["log_units"].to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# 4. Candidate-price forecasts, selection, and profit
# ---------------------------------------------------------------------------
#
# This section keeps two ideas separate:
# - method forecast: model-predicted demand used to choose a price;
# - simulator demand: structural counterfactual demand used only for evaluation.
#
# simulator_demand_at_candidate_prices creates the simulator's n-by-7 outcome
# matrix. Each row is one test observation and each column is the simulator
# demand at one of seven candidate prices, with non-price conditions held fixed.
# method_demand_indices_at_candidate_prices gives each causal method a relative
# demand curve determined only by beta_hat. A common positive demand level would
# cancel from the within-row price ranking, so it is omitted. Each predictive
# baseline instead uses all seven candidate-price predictions directly. For each
# observation, select_price_and_evaluate uses method forecasts to choose one
# price, then evaluates that fixed price against simulator demand.
#
# Key formulas:
# profit = (price - unit_cost)*demand
# change = beta * log(price_multiplier)
# simulator_demand_at_candidate_price = current_simulator_demand*exp(change)
# causal_demand_index_at_candidate_price = exp(
#     beta_hat*log(price_multiplier)
# )
# selected_option = argmax(forecast_profit across the seven price options)

# Decision settings appear only when the pricing problem is introduced. The
# +/-15% range is close to the 13.1% median non-zero price change in the retained
# M5 slice. Unit cost is generated once with each observation and remains fixed
# across candidate prices, so price choices respond to genuine margin variation.
PRICE_MULTIPLIERS = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])


def profit(
    price: np.ndarray,
    demand: np.ndarray,
    unit_cost: np.ndarray,
) -> np.ndarray:
    return (price - unit_cost) * demand


def simulator_demand_at_candidate_prices(
    test: pd.DataFrame,
    world: World,
    log_price_changes: np.ndarray,
) -> np.ndarray:
    elasticity_adjustment = world.beta * log_price_changes
    return test["simulator_demand"].to_numpy()[:, None] * np.exp(
        elasticity_adjustment
    )


def predictor_log_demand_at_candidate_prices(
    model: HistGradientBoostingRegressor | TransformerRegressor,
    test: pd.DataFrame,
) -> np.ndarray:
    log_demand_forecasts = []
    for multiplier in PRICE_MULTIPLIERS:
        candidate_features = test[PREDICTIVE_FEATURES].copy()
        candidate_features["log_price"] += np.log(multiplier)
        log_demand_forecasts.append(
            model.predict(candidate_features.to_numpy(dtype=float))
        )
    return np.column_stack(log_demand_forecasts)


def method_demand_indices_at_candidate_prices(
    models: dict[str, HistGradientBoostingRegressor | TransformerRegressor],
    test: pd.DataFrame,
    betas: dict[str, float],
) -> dict[str, np.ndarray]:
    # Causal branch: only relative demand across candidate prices matters for
    # argmax. A positive baseline demand level would multiply every candidate's
    # profit in the same row and cancel from the ranking, so it is omitted.
    relative_demand_by_method = {
        name: np.broadcast_to(
            np.exp(beta * np.log(PRICE_MULTIPLIERS))[None, :],
            (len(test), len(PRICE_MULTIPLIERS)),
        )
        for name, beta in betas.items()
    }

    # Each predictive baseline supplies both demand level and price response.
    for name, model in models.items():
        relative_demand_by_method[name] = np.exp(
            predictor_log_demand_at_candidate_prices(model, test)
        )
    return relative_demand_by_method


def select_price_and_evaluate(
    forecast_demand_by_price: np.ndarray,
    candidate_prices: np.ndarray,
    unit_cost: np.ndarray,
    simulator_demand_by_price: np.ndarray,
) -> float:
    forecast_profit_by_price = profit(
        candidate_prices,
        forecast_demand_by_price,
        unit_cost,
    )
    selected_price_index = np.argmax(forecast_profit_by_price, axis=1)
    observation_index = np.arange(len(selected_price_index))

    simulator_evaluated_profit = profit(
        candidate_prices[observation_index, selected_price_index],
        simulator_demand_by_price[observation_index, selected_price_index],
        unit_cost[:, 0],
    )
    return float(simulator_evaluated_profit.sum())


# ---------------------------------------------------------------------------
# 5. One complete world-by-seed experiment
# ---------------------------------------------------------------------------
#
# run_one connects all earlier components. It generates and splits the panel,
# trains the requested predictors, estimates the requested causal betas, builds seven candidate
# prices per test row, and obtains the simulator-demand matrix. Oracle sees all
# simulator demands at candidate prices and defines the best attainable profit
# within the common price grid. Each method's Oracle regret is its profit
# shortfall from Oracle, divided by Oracle total profit. One call returns a
# result row for each requested method.


def run_one(
    world_name: str, seed: int, n_sku: int, n_week: int, methods: list[str]
) -> list[dict]:
    world = experimental_worlds()[world_name]
    train, test = split_and_add_item_mean(generate_panel(world, seed, n_sku, n_week))
    models = {}
    if "HGB" in methods:
        models["HGB"] = fit_hgb_model(train, seed)
    if "TransformerSmall" in methods:
        models["TransformerSmall"] = fit_transformer_model(train, seed, 32, 2)
    if "TransformerLarge" in methods:
        models["TransformerLarge"] = fit_transformer_model(train, seed, 96, 3)
    if any(name in methods for name in CAUSAL_METHODS):
        betas, iv_first_stage_f = estimate_betas(train)
    else:
        betas, iv_first_stage_f = {}, np.nan

    current_price = test["price"].to_numpy()
    candidate_prices = current_price[:, None] * PRICE_MULTIPLIERS[None, :]
    unit_cost = test["unit_cost"].to_numpy()[:, None]
    log_price_changes = np.log(PRICE_MULTIPLIERS)[None, :]
    simulator_demand_by_price = simulator_demand_at_candidate_prices(
        test, world, log_price_changes
    )

    oracle_profit_by_price = profit(
        candidate_prices,
        simulator_demand_by_price,
        unit_cost,
    )
    oracle_total_profit = float(np.max(oracle_profit_by_price, axis=1).sum())
    oracle_price_index = np.argmax(oracle_profit_by_price, axis=1)
    oracle_lower_boundary_share = float(np.mean(oracle_price_index == 0))
    oracle_upper_boundary_share = float(
        np.mean(oracle_price_index == len(PRICE_MULTIPLIERS) - 1)
    )
    cost_to_current_price = test["unit_cost"].to_numpy() / current_price

    method_demand_by_price = method_demand_indices_at_candidate_prices(
        models, test, {name: beta for name, beta in betas.items() if name in methods}
    )
    rows = []
    for method in methods:
        method_total_profit = select_price_and_evaluate(
            method_demand_by_price[method],
            candidate_prices,
            unit_cost,
            simulator_demand_by_price,
        )
        beta_hat = betas.get(method, np.nan)
        rows.append(
            {
                "world": world_name,
                "world_label": world.label,
                "seed": seed,
                "method": method,
                "structural_elasticity": world.beta,
                "beta_hat": beta_hat,
                "iv_first_stage_f": iv_first_stage_f,
                "oracle_lower_boundary_share": oracle_lower_boundary_share,
                "oracle_upper_boundary_share": oracle_upper_boundary_share,
                "cost_to_current_price_median": float(
                    np.median(cost_to_current_price)
                ),
                "absolute_elasticity_error": (
                    abs(beta_hat - world.beta) if np.isfinite(beta_hat) else np.nan
                ),
                "oracle_regret": (oracle_total_profit - method_total_profit)
                / max(oracle_total_profit, 1e-9),
                "observed_wmape_pct": (
                    100
                    * np.abs(
                        np.exp(
                            models[method].predict(
                                test[PREDICTIVE_FEATURES].to_numpy(dtype=float)
                            )
                        )
                        - test["simulator_demand"].to_numpy(dtype=float)
                    ).sum()
                    / test["simulator_demand"].sum()
                    if method in models
                    else np.nan
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 6. Repeated-seed summaries and heatmap
# ---------------------------------------------------------------------------
#
# A full run creates 5 worlds x 30 seeds x 7 methods = 1,050 raw result rows.
# mean_and_ci converts repeated seed values into a mean and two-sided 95%
# interval half-width using the Student-t distribution. summarize collapses the
# seed dimension to 35 world-method rows and retains mean Oracle regret, its
# confidence interval and Monte Carlo standard error, mean estimated beta, mean
# absolute elasticity error, mean first-stage F statistic, and seed count.
# draw_heatmap pivots those rows into a world-by-method matrix. Cell text displays the
# uncapped result, while color is capped at 30% so an extreme IV failure does not
# erase the visual differences among the other methods.


def mean_and_ci(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) <= 1:
        return (float(clean.mean()) if len(clean) else np.nan, 0.0)

    t_value = float(student_t.ppf(0.975, df=len(clean) - 1))
    return float(clean.mean()), float(t_value * clean.std(ddof=1) / np.sqrt(len(clean)))


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (world, label, method), group in records.groupby(
        ["world", "world_label", "method"], sort=False
    ):
        regret, ci = mean_and_ci(100 * group["oracle_regret"])
        regret_mcse = (
            float((100 * group["oracle_regret"]).std(ddof=1) / np.sqrt(len(group)))
            if len(group) > 1
            else 0.0
        )
        beta, _ = mean_and_ci(group["beta_hat"])
        elasticity_error, _ = mean_and_ci(group["absolute_elasticity_error"])
        rows.append(
            {
                "world": world,
                "world_label": label,
                "method": method,
                "oracle_regret_pct": regret,
                "monte_carlo_ci95_half_width_pct": ci,
                "regret_mcse_pct": regret_mcse,
                "beta_hat_mean": beta,
                "absolute_elasticity_error_mean": elasticity_error,
                "iv_first_stage_f_mean": float(group["iv_first_stage_f"].mean()),
                "oracle_lower_boundary_share": float(
                    group["oracle_lower_boundary_share"].mean()
                ),
                "oracle_upper_boundary_share": float(
                    group["oracle_upper_boundary_share"].mean()
                ),
                "cost_to_current_price_median": float(
                    group["cost_to_current_price_median"].mean()
                ),
                "n_seeds": len(group),
                "observed_wmape_pct": float(group["observed_wmape_pct"].mean()),
                "n_wmape": int(group["observed_wmape_pct"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def draw_heatmap(
    summary: pd.DataFrame, output: Path, world_names: list[str], methods: list[str]
) -> None:
    worlds = experimental_worlds()
    labels = [worlds[name].label for name in world_names]
    value = summary.pivot(
        index="world_label", columns="method", values="oracle_regret_pct"
    )
    ci = summary.pivot(
        index="world_label",
        columns="method",
        values="monte_carlo_ci95_half_width_pct",
    )
    counts = summary.pivot(index="world_label", columns="method", values="n_seeds")
    value = value.loc[labels, methods]
    ci = ci.loc[labels, methods]
    counts = counts.loc[labels, methods]

    shown = value.to_numpy()
    colored = np.clip(shown, 0, 30)
    fig_height = max(3.8, 0.82 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(max(10.8, 1.6 * len(methods)), fig_height))
    image = ax.imshow(colored, cmap="RdYlGn_r", vmin=0, vmax=30, aspect="auto")

    ax.set_xticks(range(len(methods)), methods, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Method")
    ax.set_ylabel("Simulation world")
    ax.set_title("Oracle Decision Regret by Experimental World", pad=14)
    ax.set_xticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(labels)):
        for j in range(len(methods)):
            number = shown[i, j]
            text = f"{number:.0f}%" if number >= 100 else f"{number:.1f}%"
            text += f"\n+/-{ci.iloc[i, j]:.1f}"
            text += f"\nn={int(counts.iloc[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if colored[i, j] < 5 or colored[i, j] > 22 else "#222222",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    colorbar.set_label("Oracle regret (%), color capped at 30%")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Command-line entry point
# ---------------------------------------------------------------------------
#
# main reads optional world, seed, output, and quick-mode arguments. The default
# run uses 500 products and 130 weeks: exactly 104 training weeks (two annual
# cycles) followed by 26 test weeks. It runs all five worlds and seeds 0-29.
# Quick mode replaces that configuration with two representative worlds, one
# seed, 80 products, and 80 weeks. Each run_one call contributes seven records;
# the final summary is written as a CSV and rendered as a PNG heatmap.


def parse_list(text: str, cast=str) -> list:
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    worlds = experimental_worlds()
    parser = argparse.ArgumentParser(
        description="Run the full causal pricing experiment."
    )
    parser.add_argument("--quick", action="store_true", help="Fast classroom demo.")
    parser.add_argument(
        "--worlds", default=",".join(worlds), help="Comma-separated worlds."
    )
    parser.add_argument(
        "--seeds", default=",".join(map(str, range(30))), help="Comma-separated seeds."
    )
    parser.add_argument(
        "--methods", default=",".join(METHOD_NAMES),
        help="Comma-separated methods; useful for resumable predictor-only runs.",
    )
    parser.add_argument(
        "--output", default="results", help="Output folder."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reuse completed world/seed/method rows."
    )
    args = parser.parse_args()

    world_names = parse_list(args.worlds)
    world_names = [
        "hidden_confounding" if name == "stronger_hidden_confounding" else name
        for name in world_names
    ]
    seeds = parse_list(args.seeds, int)
    methods = parse_list(args.methods)
    n_sku, n_week = 500, 130
    if args.quick:
        world_names = ["baseline", "exclusion_violated"]
        seeds = [0]
        n_sku, n_week = 80, 80

    unknown = [name for name in world_names if name not in worlds]
    if unknown:
        raise SystemExit(f"Unknown worlds: {', '.join(unknown)}")
    unknown_methods = [name for name in methods if name not in METHOD_NAMES]
    if unknown_methods:
        raise SystemExit(f"Unknown methods: {', '.join(unknown_methods)}")

    output_dir = Path(__file__).resolve().parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = output_dir / "seed_results.csv"
    records = (
        pd.read_csv(raw_csv_path).to_dict("records")
        if args.resume and raw_csv_path.exists()
        else []
    )
    completed = {(row["world"], int(row["seed"]), row["method"]) for row in records}
    for world_name in world_names:
        for seed in seeds:
            pending = [name for name in methods if (world_name, seed, name) not in completed]
            if not pending:
                continue
            print(f"Running world={world_name}, seed={seed}, methods={','.join(pending)}", flush=True)
            new_rows = run_one(world_name, seed, n_sku, n_week, pending)
            records.extend(new_rows)
            completed.update((row["world"], int(row["seed"]), row["method"]) for row in new_rows)
            pd.DataFrame(records).to_csv(raw_csv_path, index=False, encoding="utf-8-sig")

    raw_results = pd.DataFrame(records)
    summary = summarize(raw_results)
    csv_path = output_dir / "summary.csv"
    image_path = output_dir / "heatmap.png"
    raw_results.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plotted_worlds = [name for name in worlds if name in set(summary["world"])]
    plotted_methods = [name for name in METHOD_NAMES if name in set(summary["method"])]
    draw_heatmap(summary, image_path, plotted_worlds, plotted_methods)

    print(
        "\n"
        + summary[
            [
                "world",
                "method",
                "oracle_regret_pct",
                "beta_hat_mean",
                "absolute_elasticity_error_mean",
                "iv_first_stage_f_mean",
                "regret_mcse_pct",
                "oracle_lower_boundary_share",
                "oracle_upper_boundary_share",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {raw_csv_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {image_path}")


if __name__ == "__main__":
    main()
