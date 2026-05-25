# Notebooks

Run these in order — each builds on the outputs of the previous one.

## 01 — Exploratory Data Analysis (`01_eda.ipynb`)

Initial exploration of the raw dataset:
- Class distribution of `TARGET_B` (confirms imbalance)
- Missing value analysis and missingness patterns
- Univariate distributions for numeric and categorical features
- Correlation analysis and feature-target relationships
- Identification of data quality issues (incoherent values, outliers)

## 02 — Feature Engineering & Selection (`02_feature_engineering_and_selection.ipynb`)

Transforms raw features into modeling-ready inputs:
- **Feature engineering**: creates domain-driven ratio and interaction features (e.g., avg gift amount, response rates, recency ratios)
- **Preprocessing scenarios**: compares three setups — raw cleaned, +feature engineering, +feature engineering + feature selection
- **Feature selection methods**: filter (correlation, mutual information), wrapper (sequential forward/backward), embedded (tree-based importance)
- **Output**: saves consensus selected features to `project_data/selected_features.json`

## 03 — Baseline Models (`03_baseline_models.ipynb`)

Compares five classifiers across the three preprocessing scenarios:
- Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN
- Evaluates each on the validation set using F1, ROC AUC, precision, recall
- Produces a summary table identifying the best scenario per model
- Results feed into notebook 04 for hyperparameter tuning

## 04 — Model Selection (`04_model_selection.ipynb`)

Hyperparameter tuning on the top-performing model–scenario combinations:
- Grid/random search with stratified cross-validation
- Threshold optimization (finds the probability cutoff that maximizes F1)
- Final model comparison and selection

## 05 — Final Submission (`05_final_submission.ipynb`)

Produces the competition submission file:
- Refits the chosen pipeline on the full training set (train + validation)
- Generates predictions on `donors_test.csv`
- Exports `project_data/Predictions.csv` in the required format (`CONTROL_NUMBER`, `TARGET_B`)

---

## Side Notebook

### 0X — Preprocessing Showcase (`0X_preprocessing_showcase.ipynb`)

Demonstrates the reusable `pipeline/` package step by step (cleaning, imputation, outlier handling, encoding, scaling). Not part of the main workflow — useful as a reference for understanding the preprocessing logic.
