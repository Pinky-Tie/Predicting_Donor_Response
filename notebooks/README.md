# Notebook Plan

This folder contains the notebook workflow for the project:

- `01_eda.ipynb` — exploratory analysis, missingness, target distribution.
- `02_feature_engineering_and_selection.ipynb` — donor-level feature engineering, preprocessing-scenario comparison, filter / wrapper / embedded feature selection, consensus list saved to `project_data/selected_features.json`.
- `03_baseline_models.ipynb` — five baseline classifiers compared across three scenarios: raw cleaned features, with feature engineering, with feature engineering + consensus feature selection. Best scenario per model feeds into the next notebook.
- `04_model_selection.ipynb` — hyper-parameter tuning and final model selection on the winning scenarios.
- `05_final_submission.ipynb` — fit the chosen pipeline on the full train and produce the competition submission file.
- `0X_preprocessing_showcase.ipynb` — side notebook demonstrating the `pipeline/` package; not part of the main numbered flow.
