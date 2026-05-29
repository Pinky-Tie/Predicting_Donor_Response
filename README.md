# Predicting Donor Response

**Data Mining II – Group Project (2025/2026)**  
Nova Information Management School – Universidade Nova de Lisboa

**Group Member Contribution**

Francisco Gomes (25%), Margarida Marchão (25%), Marta Alves (25%), Pedro Coimbras (25%)

Equally contributed to the development of the project

## Overview

Nonprofit organizations often struggle with decreasing engagement in fundraising campaigns due to donor fatigue caused by repeated and generic solicitations. The **Civic Support Alliance (CSA)** aims to modernize its outreach strategy by using data-driven methods to identify individuals most likely to respond positively to donation requests.

This project builds machine learning models to predict whether a person will donate if contacted (`TARGET_B = 1`), enabling CSA to target the right individuals while reducing unnecessary outreach. The task is framed as an **imbalanced binary classification** problem using historical campaign data.

## Original Datasets

| File | Description |
|------|-------------|
| `donors_train.csv` | Training set with 41 features + binary target (`TARGET_B`) |
| `donors_test.csv` | Test set (no target) used for final predictions |
| `sample_submission.csv` | Expected submission format: `CONTROL_NUMBER`, `TARGET_B` |

Key feature groups: donor demographics, geographic/census data, promotion and contact history, giving history, and status flags.

## Project Workflow

| # | Notebook | What it does |
|---|----------|--------------|
| 01 | `01_eda.ipynb` | Full EDA: missingness, outliers, distributions, correlation, domain-rule validation |
| 02 | `02_feature_engineering_and_selection.ipynb` | 12 engineered features, 4 preprocessing scenarios compared, filter/wrapper/embedded feature selection with consensus vote |
| 03 | `03_baseline_models.ipynb` | 5 classifiers × 3 scenarios (raw / +FE / +FE+FS); identifies LR + GB as best candidates |
| 04 | `04_model_selection.ipynb` | Optuna-based hyperparameter tuning (9 models, 100–150 trials each) |
| 05 | `05_final_submission.ipynb` | Refits best model on full train, 5-fold CV, exports predictions (5812 rows) |

See [`notebooks/README.md`](notebooks/README.md) for detailed descriptions of each notebook.

## Repository Structure

```
Predicting_Donor_Response/
├── project_data/              # Raw data, cleaned splits, and submissions
├── notebooks/                 # Ordered analysis notebooks (01–05)
├── pipeline/                  # Reusable preprocessing & modeling code
│   ├── data_loading.py
│   ├── feature_pipeline.py
│   ├── preprocessing_service.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── submission.py
│   └── preprocessing_pipeline/   # Modular steps (cleaning, imputation,
│                                  #   outliers, encoding, scaling, FE)
├── utils/                     # EDA, plotting, metric, and modeling helpers
│   ├── utils_eda.py
│   ├── utils_plots.py
│   ├── utils_metrics.py
│   └── utils_modeling.py      # Optuna wrappers, train_all_models, build_model
├── results_history.md         # Experiment log (model, params, scores)
├── requirements.txt
└── README.md
```

## Setup

### Main setup
```bash
# Clone the repository
git clone git@github.com:Pinky-Tie/Predicting_Donor_Response.git
cd Predicting_Donor_Response

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```
### Alternative setup (colab injection)
All notebooks can be easily run via the following colab notebook.
This is an easy alternative if the user's objetive is solely to retrieve fresh predictions.
[Colab Runner](https://colab.research.google.com/drive/16IblEj1HUrs2CjNJ7S-CLI2hlBcfZ0jq?usp=sharing)

## Tech Stack 

- **Python 3.13**
- pandas, NumPy — data manipulation
- scikit-learn — preprocessing, modeling, evaluation
- Optuna — hyperparameter optimization
- matplotlib, seaborn, plotly — visualization
- Jupyter Notebook — interactive analysis

## Methodology

1. **EDA** — Understand distributions, missing data patterns, class imbalance, and data quality issues.
2. **Preprocessing** — KNN imputation (numeric), most-frequent imputation (categorical), outlier rescaling, one-hot encoding. All fitting on train only to prevent leakage.
3. **Feature Engineering** — 12 domain-driven features (gift ratios, response rates, recency metrics). Evaluated across 4 preprocessing scenarios.
4. **Feature Selection** — Filter, wrapper, and embedded methods with consensus voting to produce a final feature subset.
5. **Baseline Comparison** — 5 models × 3 scenarios. Gradient Boosting achieved best ROC-AUC (~0.60); Logistic Regression best F1 (~0.39).
6. **Hyperparameter Tuning** — Optuna optimization (100–150 trials) across 9 model families. Best configuration logged to `results_history.md`.
7. **Submission** — Best model refitted on full training data, 5-fold CV for validation, predictions exported as `project_data/Predictions.csv`.

## Authors

Data Mining II group project — NOVA IMS, 2025/2026.
