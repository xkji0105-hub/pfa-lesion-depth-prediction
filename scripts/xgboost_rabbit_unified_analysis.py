"""Unified animal-level analysis of XGBoost lesion-depth predictions.

The script reproduces the XGBoost analyses reported in the manuscript:

1. An initial rabbit-level training/test split followed by rabbit-level
   bootstrap evaluation of predictive performance and SHAP importance.
2. Five-fold cross-validation grouped by rabbit.
3. A final XGBoost model fitted to all 25 rabbits for SHAP importance,
   SHAP dependence, and pairwise TreeSHAP interaction analyses.
4. Rabbit-level bootstrap confidence intervals for the Pw-Pa partial
   dependence surface, parameter-level contrasts, and Friedman H statistic.

All preprocessing is contained in a scikit-learn pipeline. Therefore, the
StandardScaler is fitted using only the data supplied to each model fit.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib import cm, colors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


FEATURE_NAMES = ["Pa", "Pw", "Pt", "d2", "d3"]
GROUP_COLUMN = "Rabbit"
TARGET_COLUMN = "Depth"

XGB_PARAMS = {
    "max_depth": 2,
    "learning_rate": 0.02,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "reg_alpha": 5,
    "reg_lambda": 15,
    "n_estimators": 1500,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "reg:squarederror",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run grouped XGBoost validation, SHAP analysis, TreeSHAP "
            "interactions, and rabbit-level bootstrap partial dependence."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("experiment_liver_data2.xlsx"),
        help="Input Excel file containing Rabbit, Pa, Pw, Pt, d2, d3, and Depth.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("xgboost_outputs"),
        help="Directory for figures and result tables.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Number of rabbit-level bootstrap iterations.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rabbits assigned to the fixed test set.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for the initial rabbit split and performance bootstrap.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=42,
        help="Random seed used by XGBoost.",
    )
    parser.add_argument(
        "--pdp-seed",
        type=int,
        default=2026,
        help="Initial random seed for the partial-dependence bootstrap.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "interpretation"],
        default="all",
        help=(
            "Use 'all' for every analysis. Use 'interpretation' to skip the "
            "initial fixed-test bootstrap while retaining cross-validation, "
            "final-model SHAP, interactions, and partial dependence."
        ),
    )
    return parser.parse_args()


def load_data(
    file_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the analysis dataset using explicit column names."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    data = pd.read_excel(file_path)
    required_columns = [GROUP_COLUMN, *FEATURE_NAMES, TARGET_COLUMN]
    missing_columns = [name for name in required_columns if name not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    analysis_data = data[required_columns].copy()
    if analysis_data[GROUP_COLUMN].isna().any():
        raise ValueError(f"{GROUP_COLUMN} contains missing values.")

    numeric_columns = [*FEATURE_NAMES, TARGET_COLUMN]
    for column in numeric_columns:
        analysis_data[column] = pd.to_numeric(
            analysis_data[column], errors="coerce"
        )
    if analysis_data[numeric_columns].isna().any().any():
        bad_columns = analysis_data[numeric_columns].columns[
            analysis_data[numeric_columns].isna().any()
        ].tolist()
        raise ValueError(f"Non-numeric or missing values found in: {bad_columns}")

    rabbit_ids = analysis_data[GROUP_COLUMN].to_numpy()
    if np.unique(rabbit_ids).size < 2:
        raise ValueError("At least two rabbits are required.")

    X = analysis_data[FEATURE_NAMES].to_numpy(dtype=float)
    y = analysis_data[TARGET_COLUMN].to_numpy(dtype=float)
    return analysis_data, rabbit_ids, X, y


def build_pipeline(random_state: int) -> Pipeline:
    """Create the StandardScaler and XGBoost pipeline used in all analyses."""
    params = dict(XGB_PARAMS)
    params["random_state"] = random_state
    return Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(**params)),
    ])


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def percentile_ci(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.percentile(values, 2.5, axis=axis),
        np.percentile(values, 97.5, axis=axis),
    )


def cluster_bootstrap_indices(
    groups: np.ndarray,
    rng: np.random.RandomState | np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample rabbits with replacement and retain every row for each draw."""
    unique_groups = np.unique(groups)
    sampled_groups = rng.choice(
        unique_groups,
        size=len(unique_groups),
        replace=True,
    )
    sampled_indices = np.concatenate([
        np.flatnonzero(groups == group_id) for group_id in sampled_groups
    ])
    return sampled_indices, sampled_groups


def build_split_summary(
    data: pd.DataFrame,
    train_rabbits: np.ndarray,
    test_rabbits: np.ndarray,
) -> pd.DataFrame:
    rows = []
    train_set = set(train_rabbits.tolist())
    test_set = set(test_rabbits.tolist())
    for rabbit_id, subset in data.groupby(GROUP_COLUMN, sort=True):
        if rabbit_id in train_set:
            split = "Train"
        elif rabbit_id in test_set:
            split = "Test"
        else:
            raise RuntimeError(f"Rabbit {rabbit_id} was not assigned to a split.")
        rows.append({
            "Rabbit": rabbit_id,
            "Split": split,
            "AblationSites": len(subset),
            "ParameterCombinations": len(subset[FEATURE_NAMES].drop_duplicates()),
        })
    return pd.DataFrame(rows)


def summarize_metrics(iteration_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize bootstrap performance using mean, SD, and percentile CI."""
    rows: list[dict[str, float | str]] = []
    for split in ["Train", "Test"]:
        for metric in ["R2", "RMSE", "MAE"]:
            values = iteration_results[f"{split}_{metric}"].to_numpy(dtype=float)
            ci_low, ci_high = np.percentile(values, [2.5, 97.5])
            rows.append({
                "Split": split,
                "Metric": metric,
                "Mean": values.mean(),
                "SD": values.std(ddof=1),
                "CI95_Low": ci_low,
                "CI95_High": ci_high,
            })
    return pd.DataFrame(rows)


def summarize_bootstrap_shap(
    shap_iterations: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Summarize mean absolute SHAP values across bootstrap model fits."""
    shap_array = shap_iterations[feature_names].to_numpy(dtype=float)
    low, high = percentile_ci(shap_array, axis=0)
    summary = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP_Mean_mm": shap_array.mean(axis=0),
        "MeanAbsSHAP_SD_mm": shap_array.std(axis=0, ddof=1),
        "CI95_Low_mm": low,
        "CI95_High_mm": high,
    })
    return summary.sort_values(
        "MeanAbsSHAP_Mean_mm", ascending=False, ignore_index=True
    )


def bootstrap_evaluation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_rabbit_ids: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    bootstrap_samples: int,
    split_seed: int,
    model_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate performance and SHAP uncertainty using rabbit bootstrap fits."""
    if bootstrap_samples < 2:
        raise ValueError("At least two bootstrap iterations are required.")

    performance_rows = []
    shap_rows = []

    for iteration in range(bootstrap_samples):
        rng = np.random.RandomState(split_seed + iteration)
        sampled_idx, sampled_rabbits = cluster_bootstrap_indices(
            train_rabbit_ids, rng
        )
        X_resampled = X_train[sampled_idx]
        y_resampled = y_train[sampled_idx]

        model = build_pipeline(random_state=model_seed)
        model.fit(X_resampled, y_resampled)

        train_metrics = metric_values(y_train, model.predict(X_train))
        test_metrics = metric_values(y_test, model.predict(X_test))
        performance_rows.append({
            "BootstrapIteration": iteration + 1,
            "SampledRabbitCount": len(sampled_rabbits),
            "UniqueSampledRabbitCount": np.unique(sampled_rabbits).size,
            "TrainingRowCount": len(sampled_idx),
            **{f"Train_{name}": value for name, value in train_metrics.items()},
            **{f"Test_{name}": value for name, value in test_metrics.items()},
        })

        # Use one fixed reference set for every bootstrap model so the SHAP
        # estimates are directly comparable. Each fitted pipeline applies the
        # scaler learned from its own resampled training data. The fixed test
        # set is not used for SHAP importance estimation.
        scaler = model.named_steps["scaler"]
        xgb_model = model.named_steps["xgb"]
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = np.asarray(
            explainer.shap_values(scaler.transform(X_train))
        )
        importance = np.abs(shap_values).mean(axis=0)
        shap_rows.append({
            "BootstrapIteration": iteration + 1,
            **dict(zip(feature_names, importance)),
        })

        if iteration == 0 or (iteration + 1) % 20 == 0:
            print(f"Bootstrap progress: {iteration + 1}/{bootstrap_samples}")

    return pd.DataFrame(performance_rows), pd.DataFrame(shap_rows)


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    model_seed: int,
) -> pd.DataFrame:
    """Perform non-randomized GroupKFold cross-validation by rabbit."""
    splitter = GroupKFold(n_splits=n_splits)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        model = build_pipeline(random_state=model_seed)
        model.fit(X[train_idx], y[train_idx])

        for split_name, idx in [("Train", train_idx), ("Test", test_idx)]:
            metrics = metric_values(y[idx], model.predict(X[idx]))
            rows.append({
                "Fold": fold,
                "Split": split_name,
                "RabbitCount": np.unique(groups[idx]).size,
                "ParameterCombinationCount": np.unique(X[idx], axis=0).shape[0],
                "AblationSiteCount": len(idx),
                **metrics,
            })

    return pd.DataFrame(rows)


def summarize_cross_validation(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize five folds with a t-based CI for the fold-wise mean."""
    t_value_df4 = 2.776445105
    rows = []
    for split_name in ["Train", "Test"]:
        split_df = cv_df[cv_df["Split"] == split_name]
        n_folds = len(split_df)
        for metric_name in ["R2", "RMSE", "MAE"]:
            values = split_df[metric_name].to_numpy(dtype=float)
            mean = values.mean()
            sd = values.std(ddof=1)
            margin = t_value_df4 * sd / np.sqrt(n_folds)
            rows.append({
                "Split": split_name,
                "Metric": metric_name,
                "Mean": mean,
                "SD": sd,
                "CI95_Low": mean - margin,
                "CI95_High": mean + margin,
            })
    return pd.DataFrame(rows)


def print_result_table(title: str, dataframe: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(dataframe.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def calculate_shap_results(
    model: Pipeline,
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Calculate full-model SHAP importance and pairwise interactions."""
    scaler = model.named_steps["scaler"]
    xgb_model = model.named_steps["xgb"]
    X_scaled = scaler.transform(X)
    explainer = shap.TreeExplainer(xgb_model)

    shap_values = np.asarray(
        explainer.shap_values(X_scaled, check_additivity=False)
    )
    interaction_values = np.asarray(
        explainer.shap_interaction_values(X_scaled)
    )

    importance = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP_mm": np.abs(shap_values).mean(axis=0),
    }).sort_values("MeanAbsSHAP_mm", ascending=False, ignore_index=True)

    pair_rows = []
    for i, j in combinations(range(len(feature_names)), 2):
        # TreeSHAP allocates half of a pairwise interaction to [i, j] and
        # half to [j, i]. Multiplication by two gives the full pair effect.
        signed_values = 2.0 * interaction_values[:, i, j]
        pair_rows.append({
            "Feature_1": feature_names[i],
            "Feature_2": feature_names[j],
            "MeanAbsInteraction_mm": np.mean(np.abs(signed_values)),
            "MeanSignedInteraction_mm": np.mean(signed_values),
        })

    interaction_df = pd.DataFrame(pair_rows).sort_values(
        "MeanAbsInteraction_mm", ascending=False, ignore_index=True
    )
    total_pair_strength = interaction_df["MeanAbsInteraction_mm"].sum()
    interaction_df["ShareOfAllPairInteractions_pct"] = (
        interaction_df["MeanAbsInteraction_mm"] / total_pair_strength * 100.0
    )
    interaction_df.insert(0, "Rank", np.arange(1, len(interaction_df) + 1))

    return shap_values, interaction_values, importance, interaction_df


def marginal_pw_pa_predictions(
    model: Pipeline,
    reference_X: np.ndarray,
    feature_names: list[str],
    pw_levels: np.ndarray,
    pa_levels: np.ndarray,
) -> np.ndarray:
    """Calculate the Pw-Pa partial-dependence surface on observed levels."""
    pw_idx = feature_names.index("Pw")
    pa_idx = feature_names.index("Pa")
    surface = np.zeros((len(pa_levels), len(pw_levels)), dtype=float)

    for pa_i, pa_value in enumerate(pa_levels):
        for pw_i, pw_value in enumerate(pw_levels):
            counterfactual_X = reference_X.copy()
            counterfactual_X[:, pa_idx] = pa_value
            counterfactual_X[:, pw_idx] = pw_value
            surface[pa_i, pw_i] = model.predict(counterfactual_X).mean()
    return surface


def bootstrap_pw_pa_pdp(
    X: np.ndarray,
    y: np.ndarray,
    rabbit_ids: np.ndarray,
    feature_names: list[str],
    pw_levels: np.ndarray,
    pa_levels: np.ndarray,
    bootstrap_samples: int,
    pdp_seed: int,
    model_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap Pw-Pa marginal predictions with rabbits as clusters."""
    bootstrap_surfaces = np.zeros(
        (bootstrap_samples, len(pa_levels), len(pw_levels)), dtype=float
    )

    for iteration in range(bootstrap_samples):
        rng = np.random.default_rng(pdp_seed + iteration)
        sampled_idx, _ = cluster_bootstrap_indices(rabbit_ids, rng)
        X_boot = X[sampled_idx]
        y_boot = y[sampled_idx]

        model = build_pipeline(random_state=model_seed)
        model.fit(X_boot, y_boot)

        # Use the current cluster-bootstrap sample as the marginalization
        # reference distribution, matching the original analysis.
        bootstrap_surfaces[iteration] = marginal_pw_pa_predictions(
            model,
            X_boot,
            feature_names,
            pw_levels,
            pa_levels,
        )

        if iteration == 0 or (iteration + 1) % 20 == 0:
            print(
                "Partial-dependence bootstrap progress: "
                f"{iteration + 1}/{bootstrap_samples}"
            )

    ci_low, ci_high = percentile_ci(bootstrap_surfaces, axis=0)
    return bootstrap_surfaces, ci_low, ci_high


def build_pdp_summary(
    central_surface: np.ndarray,
    bootstrap_surfaces: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    pw_levels: np.ndarray,
    pa_levels: np.ndarray,
) -> pd.DataFrame:
    rows = []
    boot_mean = bootstrap_surfaces.mean(axis=0)
    boot_sd = bootstrap_surfaces.std(axis=0, ddof=1)
    for pa_i, pa_value in enumerate(pa_levels):
        for pw_i, pw_value in enumerate(pw_levels):
            rows.append({
                "Pa_kV": pa_value,
                "Pw_us": pw_value,
                "FullModelMarginalPrediction_mm": central_surface[pa_i, pw_i],
                "BootstrapMean_mm": boot_mean[pa_i, pw_i],
                "BootstrapSD_mm": boot_sd[pa_i, pw_i],
                "CI95_Low_mm": ci_low[pa_i, pw_i],
                "CI95_High_mm": ci_high[pa_i, pw_i],
            })
    return pd.DataFrame(rows)


def build_effect_contrasts(
    central_surface: np.ndarray,
    bootstrap_surfaces: np.ndarray,
    pw_levels: np.ndarray,
    pa_levels: np.ndarray,
) -> pd.DataFrame:
    """Calculate all pairwise Pw and Pa marginal-prediction differences."""
    rows = []

    def add_row(
        effect: str,
        fixed_parameter: str,
        fixed_level: float,
        start_level: float,
        end_level: float,
        central_delta: float,
        bootstrap_delta: np.ndarray,
    ) -> None:
        ci_low, ci_high = percentile_ci(bootstrap_delta)
        rows.append({
            "Effect": effect,
            "FixedParameter": fixed_parameter,
            "FixedLevel": fixed_level,
            "StartLevel": start_level,
            "EndLevel": end_level,
            "FullModelDifference_mm": central_delta,
            "BootstrapMeanDifference_mm": bootstrap_delta.mean(),
            "BootstrapSD_mm": bootstrap_delta.std(ddof=1),
            "CI95_Low_mm": ci_low,
            "CI95_High_mm": ci_high,
        })

    for pa_i, pa_value in enumerate(pa_levels):
        for start_i, end_i in combinations(range(len(pw_levels)), 2):
            add_row(
                effect="Pw",
                fixed_parameter="Pa_kV",
                fixed_level=pa_value,
                start_level=pw_levels[start_i],
                end_level=pw_levels[end_i],
                central_delta=(
                    central_surface[pa_i, end_i]
                    - central_surface[pa_i, start_i]
                ),
                bootstrap_delta=(
                    bootstrap_surfaces[:, pa_i, end_i]
                    - bootstrap_surfaces[:, pa_i, start_i]
                ),
            )

    for pw_i, pw_value in enumerate(pw_levels):
        for start_i, end_i in combinations(range(len(pa_levels)), 2):
            add_row(
                effect="Pa",
                fixed_parameter="Pw_us",
                fixed_level=pw_value,
                start_level=pa_levels[start_i],
                end_level=pa_levels[end_i],
                central_delta=(
                    central_surface[end_i, pw_i]
                    - central_surface[start_i, pw_i]
                ),
                bootstrap_delta=(
                    bootstrap_surfaces[:, end_i, pw_i]
                    - bootstrap_surfaces[:, start_i, pw_i]
                ),
            )

    return pd.DataFrame(rows)


def friedman_h_from_surface(surface: np.ndarray) -> float:
    """Calculate Friedman H from a two-dimensional partial-dependence grid."""
    grand_mean = surface.mean()
    centered_joint = surface - grand_mean
    pw_main = surface.mean(axis=0) - grand_mean
    pa_main = surface.mean(axis=1) - grand_mean
    interaction_residual = (
        centered_joint - pa_main[:, None] - pw_main[None, :]
    )
    denominator = np.sum(centered_joint**2)
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.sqrt(np.sum(interaction_residual**2) / denominator))


def build_h_summary(
    central_surface: np.ndarray,
    bootstrap_surfaces: np.ndarray,
) -> pd.DataFrame:
    central_h = friedman_h_from_surface(central_surface)
    bootstrap_h = np.asarray([
        friedman_h_from_surface(surface) for surface in bootstrap_surfaces
    ])
    ci_low, ci_high = percentile_ci(bootstrap_h)
    return pd.DataFrame([{
        "Interaction": "Pa_x_Pw",
        "FullModel_Friedman_H": central_h,
        "BootstrapMean_H": bootstrap_h.mean(),
        "BootstrapSD_H": bootstrap_h.std(ddof=1),
        "CI95_Low_H": ci_low,
        "CI95_High_H": ci_high,
    }])


def plot_figure7(
    X: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list[str],
    central_surface: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    pw_levels: np.ndarray,
    pa_levels: np.ndarray,
    output_path: Path,
) -> None:
    """Plot SHAP dependence and Pw-Pa marginal predictions with 95% CIs."""
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 18,
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    })

    pa_idx = feature_names.index("Pa")
    pw_idx = feature_names.index("Pw")
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]},
    )
    ax_a, ax_b, ax_c = axes

    scatter_a = ax_a.scatter(
        X[:, pw_idx] + rng.normal(0.0, 0.035, size=len(X)),
        shap_values[:, pw_idx],
        c=X[:, pa_idx],
        cmap="viridis",
        norm=colors.Normalize(vmin=min(pa_levels), vmax=max(pa_levels)),
        s=30,
        alpha=0.72,
        linewidths=0,
    )
    ax_a.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax_a.set_xticks(pw_levels)
    ax_a.set_xlabel("Pulse Width, Pw (\u03bcs)")
    ax_a.set_ylabel("SHAP value for lesion depth (mm)")
    cbar_a = fig.colorbar(scatter_a, ax=ax_a, pad=0.02, fraction=0.052)
    cbar_a.set_label("Pulse Amplitude, Pa (kV)")

    scatter_b = ax_b.scatter(
        X[:, pa_idx] + rng.normal(0.0, 0.0025, size=len(X)),
        shap_values[:, pa_idx],
        c=X[:, pw_idx],
        cmap="viridis",
        norm=colors.Normalize(vmin=min(pw_levels), vmax=max(pw_levels)),
        s=30,
        alpha=0.72,
        linewidths=0,
    )
    ax_b.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax_b.set_xticks(pa_levels)
    ax_b.set_xlabel("Pulse Amplitude, Pa (kV)")
    ax_b.set_ylabel("SHAP value for lesion depth (mm)")
    cbar_b = fig.colorbar(scatter_b, ax=ax_b, pad=0.02, fraction=0.052)
    cbar_b.set_label("Pulse Width, Pw (\u03bcs)")

    palette = cm.viridis(np.linspace(0.15, 0.88, len(pa_levels)))
    for pa_i, (pa_value, color_value) in enumerate(zip(pa_levels, palette)):
        yerr = np.vstack([
            central_surface[pa_i] - ci_low[pa_i],
            ci_high[pa_i] - central_surface[pa_i],
        ])
        yerr = np.maximum(yerr, 0.0)
        ax_c.errorbar(
            pw_levels,
            central_surface[pa_i],
            yerr=yerr,
            color=color_value,
            marker="o",
            markersize=6,
            linewidth=1.8,
            capsize=4,
            label=f"Pa = {pa_value:.1f} kV",
        )

    ax_c.set_xticks(pw_levels)
    ax_c.set_xlabel("Pulse Width, Pw (\u03bcs)")
    ax_c.set_ylabel("Marginal predicted lesion depth (mm)")
    ax_c.legend(
        frameon=False,
        fontsize=15,
        loc="upper left",
        bbox_to_anchor=(0.10, 1.0),
    )
    ax_c.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)

    panel_labels = []
    for label, ax in zip(["(a)", "(b)", "(c)"], axes):
        text_obj = ax.text(
            0.5,
            -0.14,
            label,
            transform=ax.transAxes,
            fontsize=25,
            fontweight="bold",
            ha="center",
            va="top",
            clip_on=False,
        )
        text_obj.set_in_layout(False)
        panel_labels.append(text_obj)

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
        bbox_extra_artists=panel_labels,
        pad_inches=0.03,
        facecolor="white",
    )
    plt.close(fig)


def write_summary(
    output_path: Path,
    data: pd.DataFrame,
    cv_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    pdp_df: pd.DataFrame,
    contrast_df: pd.DataFrame,
    h_df: pd.DataFrame,
    bootstrap_samples: int,
) -> None:
    pw_pa_mask = (
        (
            (interaction_df["Feature_1"] == "Pa")
            & (interaction_df["Feature_2"] == "Pw")
        )
        | (
            (interaction_df["Feature_1"] == "Pw")
            & (interaction_df["Feature_2"] == "Pa")
        )
    )
    pw_pa_row = interaction_df.loc[pw_pa_mask].iloc[0]
    test_folds = cv_df[cv_df["Split"] == "Test"]

    lines = [
        "XGBoost grouped analysis summary",
        "",
        f"Rows: {len(data)}",
        f"Rabbits: {data[GROUP_COLUMN].nunique()}",
        f"Parameter combinations: {data[FEATURE_NAMES].drop_duplicates().shape[0]}",
        f"Rabbit-level bootstrap repetitions for PDP: {bootstrap_samples}",
        "",
        "Five-fold grouped cross-validation (test folds):",
        (
            "R2 mean +/- SD: "
            f"{test_folds['R2'].mean():.4f} +/- "
            f"{test_folds['R2'].std(ddof=1):.4f}"
        ),
        (
            "RMSE mean +/- SD: "
            f"{test_folds['RMSE'].mean():.4f} +/- "
            f"{test_folds['RMSE'].std(ddof=1):.4f} mm"
        ),
        (
            "MAE mean +/- SD: "
            f"{test_folds['MAE'].mean():.4f} +/- "
            f"{test_folds['MAE'].std(ddof=1):.4f} mm"
        ),
        "",
        "SHAP importance:",
        importance_df.to_string(index=False),
        "",
        "Pairwise SHAP interaction ranking:",
        interaction_df.to_string(index=False),
        "",
        "Pw-Pa interaction:",
        f"Rank: {int(pw_pa_row['Rank'])}",
        (
            "Mean absolute pairwise SHAP interaction: "
            f"{pw_pa_row['MeanAbsInteraction_mm']:.4f} mm"
        ),
        (
            "Share of all pairwise interaction magnitudes: "
            f"{pw_pa_row['ShareOfAllPairInteractions_pct']:.1f}%"
        ),
        "",
        "Pw-Pa marginal predictions and 95% CIs:",
        pdp_df.to_string(index=False),
        "",
        "Pw-Pa effect contrasts and 95% CIs:",
        contrast_df.to_string(index=False),
        "",
        "Pw-Pa Friedman H interaction statistic:",
        h_df.to_string(index=False),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def validate_arguments(args: argparse.Namespace, rabbit_count: int) -> None:
    if args.bootstrap < 2:
        raise ValueError("--bootstrap must be at least 2.")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    if rabbit_count < 5:
        raise ValueError("At least five rabbits are required for five-fold validation.")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, rabbit_ids, X, y = load_data(args.data)
    unique_rabbits = np.unique(rabbit_ids)
    validate_arguments(args, rabbit_count=len(unique_rabbits))

    print(f"Rows: {len(data)}")
    print(f"Rabbits: {len(unique_rabbits)}")
    print(f"Parameter combinations: {len(data[FEATURE_NAMES].drop_duplicates())}")

    split_summary = None
    bootstrap_iterations = None
    bootstrap_summary = None
    bootstrap_shap_iterations = None
    bootstrap_shap_summary = None

    if args.mode == "all":
        train_rabbits, test_rabbits = train_test_split(
            unique_rabbits,
            test_size=args.test_size,
            random_state=args.split_seed,
            shuffle=True,
        )
        train_idx = np.flatnonzero(np.isin(rabbit_ids, train_rabbits))
        test_idx = np.flatnonzero(np.isin(rabbit_ids, test_rabbits))

        if np.intersect1d(train_rabbits, test_rabbits).size:
            raise RuntimeError(
                "Rabbit-level leakage detected between train and test sets."
            )

        split_summary = build_split_summary(data, train_rabbits, test_rabbits)
        train_combinations = int(
            split_summary.loc[
                split_summary["Split"] == "Train", "ParameterCombinations"
            ].sum()
        )
        test_combinations = int(
            split_summary.loc[
                split_summary["Split"] == "Test", "ParameterCombinations"
            ].sum()
        )
        print(
            f"Initial training set: {len(train_rabbits)} rabbits, "
            f"{train_combinations} parameter combinations, "
            f"{len(train_idx)} ablation sites"
        )
        print(
            f"Fixed test set: {len(test_rabbits)} rabbits, "
            f"{test_combinations} parameter combinations, "
            f"{len(test_idx)} ablation sites"
        )

        bootstrap_iterations, bootstrap_shap_iterations = bootstrap_evaluation(
            X_train=X[train_idx],
            y_train=y[train_idx],
            train_rabbit_ids=rabbit_ids[train_idx],
            X_test=X[test_idx],
            y_test=y[test_idx],
            feature_names=FEATURE_NAMES,
            bootstrap_samples=args.bootstrap,
            split_seed=args.split_seed,
            model_seed=args.model_seed,
        )
        bootstrap_summary = summarize_metrics(bootstrap_iterations)
        bootstrap_shap_summary = summarize_bootstrap_shap(
            bootstrap_shap_iterations, FEATURE_NAMES
        )

        split_summary.to_csv(
            args.output_dir / "xgboost_animal_split.csv",
            index=False,
            encoding="utf-8-sig",
        )
        bootstrap_iterations.to_csv(
            args.output_dir / "xgboost_bootstrap_all_repetitions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        bootstrap_summary.to_csv(
            args.output_dir / "xgboost_bootstrap_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        bootstrap_shap_iterations.to_csv(
            args.output_dir / "xgboost_bootstrap_SHAP_all_repetitions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        bootstrap_shap_summary.to_csv(
            args.output_dir / "xgboost_bootstrap_SHAP_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print_result_table("Bootstrap performance summary", bootstrap_summary)
        print_result_table(
            "Bootstrap SHAP importance summary", bootstrap_shap_summary
        )

    cv_df = cross_validate_model(
        X,
        y,
        rabbit_ids,
        n_splits=5,
        model_seed=args.model_seed,
    )
    cv_summary_df = summarize_cross_validation(cv_df)
    cv_df.to_csv(
        args.output_dir / "xgboost_grouped_5fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_summary_df.to_csv(
        args.output_dir / "xgboost_grouped_5fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print_result_table("Grouped cross-validation results", cv_df)
    print_result_table("Grouped cross-validation summary", cv_summary_df)

    final_model = build_pipeline(random_state=args.model_seed)
    final_model.fit(X, y)
    shap_values, _, importance_df, interaction_df = calculate_shap_results(
        final_model,
        X,
        FEATURE_NAMES,
    )
    importance_df.to_csv(
        args.output_dir / "xgboost_SHAP_importance_full_model.csv",
        index=False,
        encoding="utf-8-sig",
    )
    interaction_df.to_csv(
        args.output_dir / "xgboost_SHAP_pairwise_interactions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print_result_table("Full-model SHAP importance", importance_df)
    print_result_table("Pairwise TreeSHAP interaction ranking", interaction_df)

    pw_levels = np.sort(data["Pw"].unique()).astype(float)
    pa_levels = np.sort(data["Pa"].unique()).astype(float)
    central_surface = marginal_pw_pa_predictions(
        final_model,
        X,
        FEATURE_NAMES,
        pw_levels,
        pa_levels,
    )
    bootstrap_surfaces, ci_low, ci_high = bootstrap_pw_pa_pdp(
        X,
        y,
        rabbit_ids,
        FEATURE_NAMES,
        pw_levels,
        pa_levels,
        bootstrap_samples=args.bootstrap,
        pdp_seed=args.pdp_seed,
        model_seed=args.model_seed,
    )
    pdp_df = build_pdp_summary(
        central_surface,
        bootstrap_surfaces,
        ci_low,
        ci_high,
        pw_levels,
        pa_levels,
    )
    contrast_df = build_effect_contrasts(
        central_surface,
        bootstrap_surfaces,
        pw_levels,
        pa_levels,
    )
    h_df = build_h_summary(central_surface, bootstrap_surfaces)

    pdp_df.to_csv(
        args.output_dir / "xgboost_Pw_Pa_PDP_rabbit_bootstrap_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contrast_df.to_csv(
        args.output_dir / "xgboost_Pw_Pa_effect_contrasts_bootstrap_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )
    h_df.to_csv(
        args.output_dir / "xgboost_Pw_Pa_Friedman_H_bootstrap_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print_result_table("Pw-Pa marginal predictions and bootstrap CIs", pdp_df)
    print_result_table("Pw-Pa contrasts and bootstrap CIs", contrast_df)
    print_result_table("Pw-Pa Friedman H and bootstrap CI", h_df)

    figure_path = args.output_dir / "Fig7_SHAP_dependence_and_bootstrap_PDP.png"
    plot_figure7(
        X,
        shap_values,
        FEATURE_NAMES,
        central_surface,
        ci_low,
        ci_high,
        pw_levels,
        pa_levels,
        figure_path,
    )

    write_summary(
        args.output_dir / "xgboost_analysis_summary.txt",
        data,
        cv_df,
        importance_df,
        interaction_df,
        pdp_df,
        contrast_df,
        h_df,
        args.bootstrap,
    )

    workbook_path = args.output_dir / "xgboost_all_analysis_results.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        if split_summary is not None:
            split_summary.to_excel(
                writer, sheet_name="Animal split", index=False
            )
            bootstrap_iterations.to_excel(
                writer, sheet_name="Bootstrap metrics all", index=False
            )
            bootstrap_summary.to_excel(
                writer, sheet_name="Bootstrap performance", index=False
            )
            bootstrap_shap_iterations.to_excel(
                writer, sheet_name="Bootstrap SHAP all", index=False
            )
            bootstrap_shap_summary.to_excel(
                writer, sheet_name="Bootstrap SHAP summary", index=False
            )
        cv_df.to_excel(writer, sheet_name="5fold metrics", index=False)
        cv_summary_df.to_excel(writer, sheet_name="5fold summary", index=False)
        importance_df.to_excel(writer, sheet_name="Full model SHAP", index=False)
        interaction_df.to_excel(
            writer, sheet_name="Pairwise interactions", index=False
        )
        pdp_df.to_excel(writer, sheet_name="Pw Pa PDP", index=False)
        contrast_df.to_excel(writer, sheet_name="Pw Pa contrasts", index=False)
        h_df.to_excel(writer, sheet_name="Friedman H", index=False)

    print(f"Analysis completed. Results saved to: {args.output_dir.resolve()}")
    print(f"Combined workbook: {workbook_path.resolve()}")


if __name__ == "__main__":
    main()
