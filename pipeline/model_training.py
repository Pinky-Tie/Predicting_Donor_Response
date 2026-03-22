# """Model selection helpers for binary donor response classification."""
# from __future__ import annotations
# import pandas as pd
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold, cross_validate
# def get_baseline_models(random_state: int = 42) -> dict[str, object]:
#     """Return a small baseline model set for comparison."""
#     return {
#         "logistic_regression": LogisticRegression(
#             max_iter=1000,
#             class_weight="balanced",
#             random_state=random_state,
#         ),
#         "random_forest": RandomForestClassifier(
#             n_estimators=300,
#             class_weight="balanced",
#             random_state=random_state,
#             n_jobs=-1,
#         ),
#         "gradient_boosting": GradientBoostingClassifier(
#             random_state=random_state,
#         ),
#     }
# def run_cross_validation(
#     model,
#     X,
#     y,
#     scoring: list[str] | None = None,
#     n_splits: int = 5,
#     random_state: int = 42,
# ) -> dict[str, float]:
#     """Run stratified cross-validation and return mean and std scores."""
#     scoring = scoring or ["roc_auc", "f1", "precision", "recall", "accuracy"]
#     cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

#     cv_results = cross_validate(
#         estimator=model,
#         X=X,
#         y=y,
#         scoring=scoring,
#         cv=cv,
#         n_jobs=-1,
#         return_train_score=False,
#     )
#     summary = {}
#     for metric in scoring:
#         summary[f"{metric}_mean"] = cv_results[f"test_{metric}"].mean()
#         summary[f"{metric}_std"] = cv_results[f"test_{metric}"].std()
#     return summary
# def compare_models(
#     models: dict[str, object],
#     X,
#     y,
#     scoring: list[str] | None = None,
#     n_splits: int = 5,
#     random_state: int = 42,
# ) -> pd.DataFrame:
#     """Compare multiple models under the same cross-validation setup."""
#     rows = []
#     for model_name, model in models.items():
#         row = {"model": model_name}
#         row.update(
#             run_cross_validation(
#                 model=model,
#                 X=X,
#                 y=y,
#                 scoring=scoring,
#                 n_splits=n_splits,
#                 random_state=random_state,
#             )
#         )
#         rows.append(row)
#     results = pd.DataFrame(rows)
#     if "roc_auc_mean" in results.columns:
#         results = results.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
#     return results

