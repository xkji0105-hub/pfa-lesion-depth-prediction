"""Animal-level bootstrap evaluation of an SVR lesion-depth model.

The initial training/test split is performed by rabbit. During each bootstrap
iteration, rabbits are sampled with replacement from the initial training
set, and every observation belonging to a selected rabbit is retained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils import resample


FEATURE_NAMES = ["Pa", "Pw", "Pt", "d2", "d3"]
GROUP_COLUMN = "Rabbit"
TARGET_COLUMN = "Depth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate support vector regression for lesion-depth prediction "
            "using an animal-level bootstrap."
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
        default=Path("svr_outputs"),
        help="Directory for result files.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Number of animal-level bootstrap iterations.",
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
        help="Random seed for the initial rabbit-level split and bootstrap.",
    )
    return parser.parse_args()


def load_data(
    file_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    data = pd.read_excel(file_path)
    required_columns = [GROUP_COLUMN, *FEATURE_NAMES, TARGET_COLUMN]
    missing_columns = [name for name in required_columns if name not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    analysis_data = data[required_columns].copy()
    if analysis_data[GROUP_COLUMN].isna().any():
        raise ValueError("Rabbit contains missing values.")

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


def build_model() -> Pipeline:
    """Create an RBF-kernel SVR with within-pipeline standardization."""
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "svr",
            SVR(
                kernel="rbf",
                C=5.0,
                gamma=0.01,
                epsilon=0.1,
                max_iter=-1,
            ),
        ),
    ])


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def summarize_metrics(iteration_results: pd.DataFrame) -> pd.DataFrame:
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


def bootstrap_evaluation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_rabbit_ids: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    bootstrap_samples: int,
    split_seed: int,
) -> pd.DataFrame:
    if bootstrap_samples < 2:
        raise ValueError("At least two bootstrap iterations are required.")

    unique_train_rabbits = np.unique(train_rabbit_ids)
    rabbit_to_indices = {
        rabbit_id: np.flatnonzero(train_rabbit_ids == rabbit_id)
        for rabbit_id in unique_train_rabbits
    }
    rows = []

    for iteration in range(bootstrap_samples):
        sampled_rabbits = resample(
            unique_train_rabbits,
            replace=True,
            n_samples=len(unique_train_rabbits),
            random_state=split_seed + iteration,
        )
        sampled_indices = np.concatenate([
            rabbit_to_indices[rabbit_id] for rabbit_id in sampled_rabbits
        ])

        model = build_model()
        model.fit(X_train[sampled_indices], y_train[sampled_indices])

        train_metrics = metric_values(y_train, model.predict(X_train))
        test_metrics = metric_values(y_test, model.predict(X_test))
        fitted_svr = model.named_steps["svr"]
        rows.append({
            "BootstrapIteration": iteration + 1,
            "SampledRabbitCount": len(sampled_rabbits),
            "UniqueSampledRabbitCount": np.unique(sampled_rabbits).size,
            "TrainingRowCount": len(sampled_indices),
            "SupportVectorCount": int(fitted_svr.support_.size),
            **{f"Train_{name}": value for name, value in train_metrics.items()},
            **{f"Test_{name}": value for name, value in test_metrics.items()},
        })

        if iteration == 0 or (iteration + 1) % 20 == 0:
            print(f"Bootstrap progress: {iteration + 1}/{bootstrap_samples}")

    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    print("\nBootstrap performance summary")
    print("=" * 78)
    for row in summary.itertuples(index=False):
        print(
            f"{row.Split:5s} {row.Metric:4s}: "
            f"{row.Mean:.4f} +/- {row.SD:.4f}; "
            f"95% CI [{row.CI95_Low:.4f}, {row.CI95_High:.4f}]"
        )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, rabbit_ids, X, y = load_data(args.data)
    unique_rabbits = np.unique(rabbit_ids)
    train_rabbits, test_rabbits = train_test_split(
        unique_rabbits,
        test_size=args.test_size,
        random_state=args.split_seed,
        shuffle=True,
    )
    train_idx = np.flatnonzero(np.isin(rabbit_ids, train_rabbits))
    test_idx = np.flatnonzero(np.isin(rabbit_ids, test_rabbits))

    if np.intersect1d(train_rabbits, test_rabbits).size:
        raise RuntimeError("Rabbit-level leakage detected between train and test sets.")

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

    print(f"Rows: {len(data)}")
    print(f"Rabbits: {len(unique_rabbits)}")
    print(f"Parameter combinations: {len(data[FEATURE_NAMES].drop_duplicates())}")
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

    iteration_results = bootstrap_evaluation(
        X_train=X[train_idx],
        y_train=y[train_idx],
        train_rabbit_ids=rabbit_ids[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        bootstrap_samples=args.bootstrap,
        split_seed=args.split_seed,
    )
    performance_summary = summarize_metrics(iteration_results)

    split_summary.to_csv(
        args.output_dir / "svr_animal_split.csv", index=False, encoding="utf-8-sig"
    )
    iteration_results.to_csv(
        args.output_dir / "svr_bootstrap_all_repetitions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    performance_summary.to_csv(
        args.output_dir / "svr_bootstrap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(performance_summary)
    print(
        "Support-vector count range: "
        f"{int(iteration_results['SupportVectorCount'].min())}-"
        f"{int(iteration_results['SupportVectorCount'].max())}"
    )
    print(f"Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
