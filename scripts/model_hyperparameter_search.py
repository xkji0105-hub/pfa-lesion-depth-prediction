"""Two-stage, rabbit-grouped hyperparameter selection for the PFA study.

The script implements the revised analysis design:

1. Split the 25 rabbits into an initial training set and a fixed test set.
2. Use only the initial training set for hyperparameter selection.
3. Keep every observation from the same rabbit in the same validation fold.
4. Run a broad randomized search followed by small sequential grid searches.
5. Lock the selected settings before evaluating the fixed test set.

MQR is the prespecified quadratic ordinary-least-squares baseline used in the
manuscript. Under that definition it has no tunable hyperparameters. The same
rabbit-grouped folds are still used to estimate its training-set CV RMSE.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform, uniform
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - handled with a clear runtime message
    XGBRegressor = None


FEATURE_NAMES = ["Pa", "Pw", "Pt", "d2", "d3"]
GROUP_COLUMN = "Rabbit"
TARGET_COLUMN = "Depth"
MODEL_NAMES = ["MQR", "BPNN", "SVR", "XGBoost"]


@dataclass(frozen=True)
class SearchPlan:
    estimator: BaseEstimator
    random_space: dict[str, Any]
    random_iterations: int
    fine_grid_builder: Callable[[dict[str, Any]], list[tuple[str, dict[str, list[Any]]]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select BPNN, SVR, and XGBoost hyperparameters using only the "
            "initial training rabbits. Evaluate MQR as a fixed quadratic OLS "
            "baseline."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Excel file containing Rabbit, Pa, Pw, Pt, d2, d3, and Depth.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hyperparameter_search_outputs"),
        help="Directory for search results and selected parameters.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help="Models to run. The default runs all four models.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rabbits assigned to the fixed test set.",
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=5,
        help="Number of rabbit-grouped folds used within the training set.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for the initial rabbit-level split.",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=42,
        help="Random seed for randomized search and stochastic models.",
    )
    parser.add_argument(
        "--search-jobs",
        type=int,
        default=-1,
        help="Parallel jobs used by scikit-learn searches. Use 1 if memory is limited.",
    )
    parser.add_argument(
        "--random-iterations",
        type=int,
        default=None,
        help=(
            "Optional override for randomized-search iterations in every tunable "
            "model. By default, the script uses 60 for BPNN/SVR and 80 for XGBoost."
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity passed to RandomizedSearchCV and GridSearchCV.",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    data = pd.read_excel(path)
    required = [GROUP_COLUMN, *FEATURE_NAMES, TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data[required].copy()
    if data[GROUP_COLUMN].isna().any():
        raise ValueError(f"{GROUP_COLUMN} contains missing values.")
    for column in [*FEATURE_NAMES, TARGET_COLUMN]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    bad = data[[*FEATURE_NAMES, TARGET_COLUMN]].columns[
        data[[*FEATURE_NAMES, TARGET_COLUMN]].isna().any()
    ].tolist()
    if bad:
        raise ValueError(f"Non-numeric or missing values found in: {bad}")
    if data[GROUP_COLUMN].nunique() < 2:
        raise ValueError("At least two rabbits are required.")
    return data


def split_by_rabbit(
    data: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rabbits = np.asarray(sorted(pd.unique(data[GROUP_COLUMN]).tolist()))
    train_rabbits, test_rabbits = train_test_split(
        rabbits,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    train_mask = data[GROUP_COLUMN].isin(train_rabbits).to_numpy()
    test_mask = data[GROUP_COLUMN].isin(test_rabbits).to_numpy()
    if np.any(train_mask & test_mask):
        raise RuntimeError("Rabbit-level leakage detected between train and test sets.")

    rows = []
    for split_name, mask in [("Initial training", train_mask), ("Fixed test", test_mask)]:
        subset = data.loc[mask]
        rows.append(
            {
                "Split": split_name,
                "RabbitCount": int(subset[GROUP_COLUMN].nunique()),
                "ParameterCombinationCount": int(
                    subset[FEATURE_NAMES].drop_duplicates().shape[0]
                ),
                "AblationSiteCount": int(len(subset)),
            }
        )
    return train_mask, test_mask, pd.DataFrame(rows)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def test_metrics(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    prediction = estimator.predict(X)
    return {
        "R2": float(r2_score(y, prediction)),
        "RMSE": rmse(y, prediction),
        "MAE": float(mean_absolute_error(y, prediction)),
    }


def rounded_unique(values: list[float], lower: float, upper: float) -> list[float]:
    clipped = [min(max(float(value), lower), upper) for value in values]
    return sorted({round(value, 8) for value in clipped})


def multiplicative_grid(
    center: float,
    lower: float,
    upper: float,
    factors: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> list[float]:
    return rounded_unique([center * factor for factor in factors], lower, upper)


def bpnn_fine_grids(best: dict[str, Any]) -> list[tuple[str, dict[str, list[Any]]]]:
    width = int(best["model__hidden_layer_sizes"][0])
    widths = sorted({max(2, width - 5), width, width + 5})
    return [
        ("architecture", {"model__hidden_layer_sizes": [(value,) for value in widths]}),
        (
            "optimization",
            {
                "model__learning_rate_init": multiplicative_grid(
                    float(best["model__learning_rate_init"]), 1e-5, 0.2
                ),
                "model__alpha": multiplicative_grid(
                    float(best["model__alpha"]), 1e-6, 100.0
                ),
            },
        ),
    ]


def svr_fine_grids(best: dict[str, Any]) -> list[tuple[str, dict[str, list[Any]]]]:
    return [
        (
            "kernel_scale",
            {
                "model__C": multiplicative_grid(float(best["model__C"]), 1e-3, 1e4),
                "model__gamma": multiplicative_grid(
                    float(best["model__gamma"]), 1e-6, 100.0
                ),
            },
        ),
        (
            "epsilon",
            {
                "model__epsilon": multiplicative_grid(
                    float(best["model__epsilon"]), 1e-4, 2.0,
                    factors=(0.75, 1.0, 1.25),
                )
            },
        ),
    ]


def xgboost_fine_grids(best: dict[str, Any]) -> list[tuple[str, dict[str, list[Any]]]]:
    trees = int(best["model__n_estimators"])
    depth = int(best["model__max_depth"])
    return [
        (
            "boosting",
            {
                "model__n_estimators": sorted(
                    {max(50, trees - 250), trees, min(3000, trees + 250)}
                ),
                "model__max_depth": sorted({max(1, depth - 1), depth, min(10, depth + 1)}),
                "model__learning_rate": multiplicative_grid(
                    float(best["model__learning_rate"]),
                    0.001,
                    0.5,
                    factors=(0.75, 1.0, 1.25),
                ),
            },
        ),
        (
            "sampling",
            {
                "model__subsample": rounded_unique(
                    [float(best["model__subsample"]) - 0.1,
                     float(best["model__subsample"]),
                     float(best["model__subsample"]) + 0.1],
                    0.3,
                    1.0,
                ),
                "model__colsample_bytree": rounded_unique(
                    [float(best["model__colsample_bytree"]) - 0.1,
                     float(best["model__colsample_bytree"]),
                     float(best["model__colsample_bytree"]) + 0.1],
                    0.3,
                    1.0,
                ),
            },
        ),
        (
            "regularization",
            {
                "model__reg_alpha": multiplicative_grid(
                    float(best["model__reg_alpha"]), 1e-6, 100.0
                ),
                "model__reg_lambda": multiplicative_grid(
                    float(best["model__reg_lambda"]), 1e-6, 100.0
                ),
            },
        ),
    ]


def build_search_plans(seed: int) -> dict[str, SearchPlan]:
    bpnn = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    solver="adam",
                    max_iter=5000,
                    tol=1e-6,
                    random_state=seed,
                ),
            ),
        ]
    )
    svr = Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))])

    plans = {
        "BPNN": SearchPlan(
            estimator=bpnn,
            random_space={
                "model__hidden_layer_sizes": [(5,), (10,), (15,), (20,), (30,), (50,)],
                "model__activation": ["relu", "tanh"],
                "model__learning_rate_init": loguniform(1e-4, 5e-2),
                "model__alpha": loguniform(1e-4, 10.0),
                "model__batch_size": [16, 32, 64, "auto"],
                "model__learning_rate": ["constant", "adaptive"],
            },
            random_iterations=60,
            fine_grid_builder=bpnn_fine_grids,
        ),
        "SVR": SearchPlan(
            estimator=svr,
            random_space={
                "model__C": loguniform(1e-2, 1e3),
                "model__gamma": loguniform(1e-5, 10.0),
                "model__epsilon": loguniform(1e-3, 1.0),
            },
            random_iterations=60,
            fine_grid_builder=svr_fine_grids,
        ),
    }

    if XGBRegressor is not None:
        xgb = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        plans["XGBoost"] = SearchPlan(
            estimator=xgb,
            random_space={
                "model__n_estimators": randint(100, 2001),
                "model__max_depth": randint(1, 7),
                "model__learning_rate": loguniform(0.005, 0.2),
                "model__subsample": uniform(0.4, 0.6),
                "model__colsample_bytree": uniform(0.4, 0.6),
                "model__reg_alpha": loguniform(1e-4, 20.0),
                "model__reg_lambda": loguniform(1e-3, 30.0),
            },
            random_iterations=80,
            fine_grid_builder=xgboost_fine_grids,
        )
    return plans


def mqr_estimator() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("quadratic", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression(fit_intercept=True)),
        ]
    )


def save_cv_results(search: Any, path: Path) -> None:
    results = pd.DataFrame(search.cv_results_)
    columns = [
        column
        for column in results.columns
        if column.startswith("param_")
        or column
        in {
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "mean_train_score",
            "std_train_score",
            "mean_fit_time",
            "std_fit_time",
        }
    ]
    results[columns].sort_values("rank_test_score").to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def run_tunable_model(
    model_name: str,
    plan: SearchPlan,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    seed: int,
    n_jobs: int,
    verbose: int,
    random_iterations_override: int | None,
) -> tuple[BaseEstimator, float, dict[str, Any]]:
    model_dir = output_dir / model_name.lower()
    model_dir.mkdir(parents=True, exist_ok=True)

    random_search = RandomizedSearchCV(
        estimator=clone(plan.estimator),
        param_distributions=plan.random_space,
        n_iter=(
            random_iterations_override
            if random_iterations_override is not None
            else plan.random_iterations
        ),
        scoring="neg_root_mean_squared_error",
        cv=cv_splits,
        random_state=seed,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
        error_score="raise",
        verbose=verbose,
    )
    random_search.fit(X_train, y_train)
    save_cv_results(random_search, model_dir / "random_search_results.csv")

    current_params = dict(random_search.best_params_)
    current_estimator = clone(plan.estimator).set_params(**current_params)
    stage_records = [
        {
            "Stage": "random_search",
            "BestInnerCV_RMSE": float(-random_search.best_score_),
            "SelectedParameters": json.dumps(current_params, default=str, sort_keys=True),
        }
    ]

    for order, (stage_name, grid) in enumerate(plan.fine_grid_builder(current_params), start=1):
        grid_search = GridSearchCV(
            estimator=current_estimator,
            param_grid=grid,
            scoring="neg_root_mean_squared_error",
            cv=cv_splits,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=True,
            error_score="raise",
            verbose=verbose,
        )
        grid_search.fit(X_train, y_train)
        save_cv_results(
            grid_search,
            model_dir / f"grid_{order}_{stage_name}_results.csv",
        )
        current_params.update(grid_search.best_params_)
        current_estimator = clone(plan.estimator).set_params(**current_params)
        stage_records.append(
            {
                "Stage": f"grid_{order}_{stage_name}",
                "BestInnerCV_RMSE": float(-grid_search.best_score_),
                "SelectedParameters": json.dumps(
                    current_params, default=str, sort_keys=True
                ),
            }
        )

    pd.DataFrame(stage_records).to_csv(
        model_dir / "search_stage_summary.csv", index=False, encoding="utf-8-sig"
    )
    final_estimator = clone(plan.estimator).set_params(**current_params)
    final_scores = cross_val_score(
        final_estimator,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv_splits,
        n_jobs=n_jobs,
        error_score="raise",
    )
    final_estimator.fit(X_train, y_train)
    return final_estimator, float(-final_scores.mean()), current_params


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.inner_folds < 2:
        raise ValueError("--inner-folds must be at least 2.")
    if args.random_iterations is not None and args.random_iterations < 1:
        raise ValueError("--random-iterations must be at least 1.")
    if "XGBoost" in args.models and XGBRegressor is None:
        raise ImportError("xgboost is required when XGBoost is selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data)
    train_mask, test_mask, split_summary = split_by_rabbit(
        data, test_size=args.test_size, seed=args.split_seed
    )
    split_summary.to_csv(
        args.output_dir / "data_split_summary.csv", index=False, encoding="utf-8-sig"
    )

    X = data[FEATURE_NAMES].to_numpy(dtype=float)
    y = data[TARGET_COLUMN].to_numpy(dtype=float)
    groups = data[GROUP_COLUMN].to_numpy()
    X_train, y_train, groups_train = X[train_mask], y[train_mask], groups[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if np.unique(groups_train).size < args.inner_folds:
        raise ValueError("The training set has fewer rabbits than --inner-folds.")
    splitter = GroupKFold(n_splits=args.inner_folds)
    cv_splits = list(splitter.split(X_train, y_train, groups_train))
    for train_index, validation_index in cv_splits:
        overlap = np.intersect1d(groups_train[train_index], groups_train[validation_index])
        if overlap.size:
            raise RuntimeError(f"Rabbit-level leakage detected within CV: {overlap}")

    config = {
        "data": str(args.data.resolve()),
        "features": FEATURE_NAMES,
        "target": TARGET_COLUMN,
        "group": GROUP_COLUMN,
        "test_size": args.test_size,
        "inner_folds": args.inner_folds,
        "split_seed": args.split_seed,
        "search_seed": args.search_seed,
        "random_iterations_override": args.random_iterations,
        "selection_metric": "mean rabbit-grouped CV RMSE on initial training set",
        "test_set_used_during_tuning": False,
        "models": args.models,
    }
    (args.output_dir / "analysis_configuration.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plans = build_search_plans(args.search_seed)
    summary_rows = []
    selected_parameters: dict[str, Any] = {}

    for model_name in args.models:
        print(f"\n{'=' * 80}\n{model_name}\n{'=' * 80}")
        if model_name == "MQR":
            estimator = mqr_estimator()
            scores = cross_val_score(
                estimator,
                X_train,
                y_train,
                scoring="neg_root_mean_squared_error",
                cv=cv_splits,
                n_jobs=args.search_jobs,
                error_score="raise",
            )
            inner_rmse = float(-scores.mean())
            estimator.fit(X_train, y_train)
            params: dict[str, Any] = {
                "quadratic__degree": 2,
                "quadratic__include_bias": False,
                "model": "ordinary least squares",
            }
            procedure = "Fixed quadratic OLS specification; no tunable hyperparameters"
        else:
            estimator, inner_rmse, params = run_tunable_model(
                model_name=model_name,
                plan=plans[model_name],
                X_train=X_train,
                y_train=y_train,
                cv_splits=cv_splits,
                output_dir=args.output_dir,
                seed=args.search_seed,
                n_jobs=args.search_jobs,
                verbose=args.verbose,
                random_iterations_override=args.random_iterations,
            )
            procedure = "Randomized search followed by sequential fine grid search"

        metrics = test_metrics(estimator, X_test, y_test)
        selected_parameters[model_name] = params
        summary_rows.append(
            {
                "Model": model_name,
                "SelectionProcedure": procedure,
                "InnerCV_RMSE": inner_rmse,
                "FixedTest_R2": metrics["R2"],
                "FixedTest_RMSE": metrics["RMSE"],
                "FixedTest_MAE": metrics["MAE"],
                "SelectedParameters": json.dumps(params, default=str, sort_keys=True),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("InnerCV_RMSE")
    summary.to_csv(
        args.output_dir / "model_selection_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "selected_parameters.json").write_text(
        json.dumps(selected_parameters, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print("\nModel selection summary")
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
