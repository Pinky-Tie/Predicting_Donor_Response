# Notebooks

Run these in order — each builds on the outputs of the previous one.

## 01 — Exploratory Data Analysis (`01_eda.ipynb`)

Full exploration of the raw dataset:
- Summary statistics and data types for all 41 features
- Missingness analysis (percentages, overlap heatmap, missing vs target relationship)
- Outlier detection (boxplots, IQR-based percentages, histograms)
- Categorical frequency analysis and target-rate plots
- Correlation heatmap
- Domain-rule validation (negative CHILDREN, proportions > 1, "?" codes in SES/URBANICITY)
- Ends with preprocessing recommendations based on findings

## 02 — Feature Engineering & Selection (`02_feature_engineering_and_selection.ipynb`)

Transforms raw features into modeling-ready inputs:
- 70/30 stratified train/val split (`random_state=5`)
- Full preprocessing via `preprocess_data()` (KNN imputation, outlier rescaling, one-hot encoding) applied to train, val, and test
- **12 engineered features** created (AVG_GIFT_PER_DONATION, PROMOTION_RESPONSE_RATE, RESPONSES_PER_YEAR, etc.)
- Baseline comparison: Logistic Regression + Gradient Boosting evaluated across 4 preprocessing scenarios (drop/keep WEALTH_RATING × numeric/categorical INCOME_GROUP)
- Comparison of baseline vs engineered feature sets (16-row results table)
- **Feature selection**: filter (correlation, mutual information), wrapper (sequential), embedded (tree-based importance) — consensus vote
- **Output**: `project_data/selected_features.json`, cleaned CSVs (`X_train_clean.csv`, `X_val_clean.csv`, `X_test_clean.csv`, `y_train_clean.csv`, `y_val_clean.csv`)
- **Key finding**: engineered features did NOT improve Gradient Boosting but marginally helped Logistic Regression

## 03 — Baseline Models (`03_baseline_models.ipynb`)

Compares 5 classifiers across 3 preprocessing scenarios:
- **Models**: Logistic Regression, Random Forest, Decision Tree, Extra Trees, Gradient Boosting
- **Scenarios**: Raw (cleaned only), +FE (with engineered features), +FE+FS (engineered + consensus feature selection)
- Metrics: ROC-AUC, F1, precision, recall, accuracy (train and validation)
- Side-by-side scenario comparison table
- **Key results**: Gradient Boosting best ROC-AUC (~0.60), Logistic Regression best F1 (~0.39). All models struggle with recall at default threshold.
- **Conclusion**: recommends threshold tuning; identifies LR + GB as candidates for hyperparameter tuning

## 04 — Model Selection (`04_model_selection.ipynb`)

Hyperparameter tuning framework using **Optuna**:
- Defines optimization wrappers for selected models, considering the highest performers in the previous notebook.
- Uses `utils/utils_modeling.py` (`optimize_with_optuna`, `train_all_models`)
- Designed to run 100–150 trials per model and log best results to `results_history.md`

## 05 — Final Submission (`05_final_submission.ipynb`)

Produces the competition submission:
- Builds the model the winner model defined in the previous notebook
- 5-fold cross-validation on full training data (train + val)
- Final fit and validation F1 reported
- Predicts on `donors_test.csv` (5812 rows)
- **Output**: `project_data/Predictions.csv` (`CONTROL_NUMBER`, `TARGET_B`)

