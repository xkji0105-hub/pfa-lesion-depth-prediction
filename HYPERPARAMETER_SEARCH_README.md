# Rabbit grouped hyperparameter search

`model_hyperparameter_search.py` implements the tuning procedure for the revised manuscript.

## Analysis design

- The initial split is performed by rabbit.
- Randomized search and fine grid search use only the initial training set.
- Inner cross-validation is grouped by rabbit.
- Mean validation RMSE is the selection criterion.
- The fixed test set is evaluated only after the parameters have been selected.
- MQR remains the prespecified quadratic ordinary-least-squares model. It has no tunable hyperparameters under the definition used in the manuscript.

The script records the animal split, every random-search result, every grid-search result, the selected parameters, inner cross-validation RMSE, and fixed-test metrics.

## Required columns

The Excel input file must contain:

```text
Rabbit, Pa, Pw, Pt, d2, d3, Depth
```

## Dependencies

```text
numpy
pandas
scipy
scikit-learn
xgboost
openpyxl
```

Use the same package versions as the main analysis repository when reproducing manuscript results.

## Run all models

```powershell
python model_hyperparameter_search.py `
  --data "C:\path\to\experiment_liver_data2.xlsx" `
  --output-dir "C:\path\to\tuning_results" `
  --models MQR BPNN SVR XGBoost `
  --inner-folds 5 `
  --split-seed 42 `
  --search-seed 42
```

If memory is limited, add `--search-jobs 1`.

For a short installation test, add `--random-iterations 2`. Do not use this
reduced value for the manuscript analysis.

## Main outputs

- `data_split_summary.csv`: rabbit, parameter-combination, and ablation-site counts.
- `analysis_configuration.json`: split, seeds, folds, features, and selection metric.
- `model_selection_summary.csv`: inner-CV and fixed-test performance.
- `selected_parameters.json`: final settings for all models.
- `<model>/random_search_results.csv`: all randomized-search candidates.
- `<model>/grid_*_results.csv`: all fine-grid candidates.
- `<model>/search_stage_summary.csv`: best result after each search stage.


