# Prediction of PFA Lesion Depth

This repository contains the Python code used to evaluate regression models
for predicting pulsed field ablation (PFA) lesion depth from combinations of
pulse parameters in rabbit liver experiments.

The analysis was developed for a specific experimental configuration involving
rabbit liver tissue, a bipolar linear ablation catheter, a standardized contact
procedure, and a biphasic pulse waveform. The resulting models and parameter
importance estimates should not be assumed to generalize directly to other
tissues, catheters, pulse generators, or waveform configurations.

## Models

Four regression models are included:

- Multiple quadratic regression (MQR)
- Backpropagation neural network (BPNN)
- Support vector regression (SVR)
- eXtreme Gradient Boosting (XGBoost)

The MQR, BPNN, and SVR scripts perform model evaluation using an initial split
at the animal level followed by rabbit-level bootstrap resampling. The unified
XGBoost script additionally performs grouped five-fold cross-validation, SHAP
analysis, pairwise TreeSHAP interaction analysis, two-dimensional partial
dependence analysis, and calculation of the Friedman H interaction statistic.

## Repository structure

```text
pfa-lesion-depth-prediction/
|-- README.md
|-- requirements.txt
|-- LICENSE
|-- .gitignore
`-- scripts/
    |-- mqr_rabbit_bootstrap.py
    |-- bpnn_rabbit_bootstrap.py
    |-- svr_rabbit_bootstrap.py
    `-- xgboost_rabbit_unified_analysis.py
```

The experimental dataset is not distributed in this repository. Users must
provide a local Excel file with the structure described below.

## Software environment

The analyses were verified using Python 3.10 and the package versions listed in
`requirements.txt`.

Create and activate a virtual environment before installing the dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Required input data

The input file must be an Excel workbook containing the following columns with
these exact names:

| Column | Description | Unit |
|---|---|---|
| `Rabbit` | Rabbit identifier used for grouped splitting and resampling | None |
| `Pa` | Pulse amplitude | kV |
| `Pw` | Pulse width | microseconds |
| `Pt` | Number of pulse trains | None |
| `d2` | Inter-pulse delay | microseconds |
| `d3` | Pulse-train interval | seconds |
| `Depth` | Measured lesion depth | mm |

The P2 dataset used in the study contained 432 lesion measurements from 25
rabbits and 72 pulse-parameter combinations. Each parameter combination had six
ablation sites. The six measurements belonging to the same parameter
combination were obtained from the same liver lobe and were not treated as
independent experimental units when splitting or resampling the data.

## Analysis design

For the bootstrap model comparison, the data are initially split by rabbit:

- 20 rabbits form the initial training set.
- Five rabbits form a fixed test set.
- Twenty rabbits are sampled with replacement from the initial training set in
  each bootstrap iteration.
- Every observation belonging to a sampled rabbit is retained.
- The fitted model is evaluated on the same fixed test animals in every
  bootstrap iteration.
- The default number of bootstrap iterations is 200.

All standardization steps are contained within the model pipelines. The scaling
parameters are therefore estimated only from the data used for the corresponding
model fit and are then applied to the relevant evaluation data.

The XGBoost script also performs five-fold `GroupKFold`
cross-validation. All measurements from the same rabbit remain within one fold,
and every rabbit appears in a test fold exactly once.

## Running the models

Replace `path/to/experiment_liver_data2.xlsx` with the actual local path to the
Excel file. Output directories are created automatically.

### MQR

```bash
python scripts/mqr_rabbit_bootstrap.py --data "path/to/experiment_liver_data2.xlsx" --output-dir "outputs/mqr" --bootstrap 200
```

### BPNN

```bash
python scripts/bpnn_rabbit_bootstrap.py --data "path/to/experiment_liver_data2.xlsx" --output-dir "outputs/bpnn" --bootstrap 200
```

### SVR

```bash
python scripts/svr_rabbit_bootstrap.py --data "path/to/experiment_liver_data2.xlsx" --output-dir "outputs/svr" --bootstrap 200
```

### XGBoost

```bash
python scripts/xgboost_rabbit_unified_analysis.py --data "path/to/experiment_liver_data2.xlsx" --output-dir "outputs/xgboost" --bootstrap 200 --mode all
```

The XGBoost script also supports `--mode interpretation`, which skips the
initial fixed-test bootstrap while retaining grouped cross-validation, final
model SHAP analysis, interaction analysis, and partial dependence analysis.

Use the `--help` option to display all available arguments. For example:

```bash
python scripts/xgboost_rabbit_unified_analysis.py --help
```

## Main outputs

The MQR, BPNN, and SVR scripts generate:

- The animal-level training and test split
- Metrics from every bootstrap iteration
- Mean, standard deviation, and percentile 95% confidence interval for R2,
  RMSE, and MAE

The unified XGBoost script additionally generates:

- Bootstrap uncertainty estimates for SHAP importance
- Metrics from grouped five-fold cross-validation
- SHAP importance estimates from the final model
- Pairwise TreeSHAP interaction values
- Pw-Pa marginal predictions with rabbit-level bootstrap confidence intervals
- Pairwise differences between Pw and Pa levels
- The Friedman H statistic and its bootstrap confidence interval
- The combined SHAP dependence and partial dependence figure
- A combined Excel workbook containing all result tables

## Reproducibility

The default random seeds reproduce the analyses reported in the manuscript:

- Initial animal-level split and performance bootstrap: 42
- XGBoost model: 42
- Partial dependence bootstrap: 2026

The grouped five-fold cross-validation uses `GroupKFold(n_splits=5)` without
shuffling. Changing a random seed, model parameter, data file, or package
version may change the numerical results.

## Data availability

The experimental data are not included in this repository. The code and the
required data schema are provided to document the analysis workflow and support
reproducibility when the corresponding data are available.

## Citation

If this code is used in another study, please cite the associated manuscript:

> Prediction of Pulsed Field Ablation Lesion Depth Using a Linear Ablation
> Catheter with Different Pulse Parameter Combinations: An Animal Study.

The complete journal citation will be added after publication.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
