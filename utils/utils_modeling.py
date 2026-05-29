"""Importing the libraries needed for data handling, preprocessing, modeling, and evaluation"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from IPython.display import display
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)   # keep output clean

training_scores = []
val_scores = []


# ── Objective ─────────────────────────────────────────────────────────────────

def objective(train_data, y_train, val_data, y_val, trial, model_type):
    """
    Build a model from a `model_type` and evaluate one Optuna trial.

    Parameters
    ----------
    train_data : array-like or DataFrame
        Feature matrix used for cross-validation and training.
    y_train : array-like
        Target labels for training.
    val_data : array-like or DataFrame
        Holdout validation features (used for final holdout score reporting).
    y_val : array-like
        Holdout validation labels.
    trial : optuna.trial.Trial
        Optuna trial object used to suggest hyperparameters.
    model_type : str
        Short name indicating which model to build (e.g. "RandomForestClassifier").

    Returns
    -------
    float
        Mean cross-validation F1 score (this is what Optuna maximizes).
    """

    if model_type == "GradientBoosting":
        model = GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 400),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 10, 50),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=1,
        )

    elif model_type == "LogisticRegression":
        model = LogisticRegression(
            C=trial.suggest_float("C", 0.001, 10.0, log=True),
            solver=trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
            max_iter=trial.suggest_int("max_iter", 200, 1000),
            fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
            class_weight="balanced",
            random_state=1,
        )

    elif model_type == "AdaBoost":
        base = DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 1, 4),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 8),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            class_weight="balanced",
            random_state=1,
        )
        model = AdaBoostClassifier(
            estimator=base,
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            algorithm="SAMME",
            random_state=14,
        )

    elif model_type == "DecisionTree":
        model = DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 30),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight="balanced",
            random_state=1,
        )

    elif model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 15),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 10, 100),
            class_weight="balanced",
            random_state=1,
        )

    elif model_type == "GaussianNB":
        model = GaussianNB(
            var_smoothing=10 ** trial.suggest_float("var_smoothing", -9, -1)
        )

    elif model_type == "KNeighborsClassifier":
        model = KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 3, 30),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            algorithm=trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
            leaf_size=trial.suggest_int("leaf_size", 10, 50),
            p=trial.suggest_int("p", 1, 2),
        )

    elif model_type == "MLPClassifier":
        model = MLPClassifier(
            hidden_layer_sizes=trial.suggest_categorical(
                "hidden_layer_sizes", [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)]
            ),
            activation=trial.suggest_categorical("activation", ["logistic", "tanh", "relu"]),
            solver=trial.suggest_categorical("solver", ["lbfgs", "adam"]),
            learning_rate=trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
            early_stopping=True,
            max_iter=trial.suggest_int("max_iter", 200, 1000),
            random_state=1,
        )

    elif model_type == "ExtraTreesClassifier":
        model = ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 15),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 10, 100),
            class_weight="balanced",
            random_state=1,
        )

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    # ── Cross-validation score (what Optuna maximises) ────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    if model_type == "GradientBoosting":
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        sample_weights = np.where(y_train == 1, neg / pos, 1.0).ravel()  # ← .ravel() forces 1D
        cv_scores = cross_val_score(
            model, train_data, y_train, cv=cv, scoring="f1",
            params={"sample_weight": sample_weights},
        )
    else:
        cv_scores = cross_val_score(model, train_data, y_train, cv=cv, scoring="f1")

    validation_score = cv_scores.mean()

    # ── Training score (logged for the convergence plot) ──────────────────────
    if model_type == "GradientBoosting":
        model.fit(train_data, y_train, sample_weight=sample_weights)
    else:
        model.fit(train_data, y_train)

    training_score = f1_score(y_train, model.predict(train_data))

    training_scores.append(training_score)
    val_scores.append(validation_score)

    return validation_score

# ── Build a model from best params ────────────────────────────────────────────

def build_model(model_type, params):
    """
    Reconstruct an unfitted scikit-learn estimator from a parameter dict.

    Parameters
    ----------
    model_type : str
        Name of the model type (must be one of the keys handled by the mapping).
    params : dict
        Dictionary of hyperparameters to pass to the estimator constructor.

    Returns
    -------
    estimator
        A scikit-learn estimator instance constructed with the given params.

    Raises
    ------
    ValueError
        If `model_type` is unknown.
    """
    mapping = {
        "GradientBoosting":       lambda p: GradientBoostingClassifier(**p, random_state=1),
        "LogisticRegression":     lambda p: LogisticRegression(**p, random_state=1),
        "AdaBoost":               lambda p: AdaBoostClassifier(**p, random_state=1),
        "DecisionTree":           lambda p: DecisionTreeClassifier(**p, class_weight="balanced", random_state=1),
        "RandomForestClassifier": lambda p: RandomForestClassifier(**p, class_weight="balanced", random_state=1),
        "GaussianNB":             lambda p: GaussianNB(**p),
        "KNeighborsClassifier":   lambda p: KNeighborsClassifier(**p),
        "MLPClassifier":          lambda p: MLPClassifier(**p, random_state=1),
        "ExtraTreesClassifier":   lambda p: ExtraTreesClassifier(**p, class_weight="balanced", random_state=1),
    }
    if model_type not in mapping:
        raise ValueError(f"Unknown model_type: '{model_type}'")
    return mapping[model_type](params)


# ── Main optimisation routine ─────────────────────────────────────────────────

def optimize_with_optuna(
    model_type, train_data, y_train, val_data, y_val,
    n_trials=100, hyper_importance=False, slice_plot=False, optimization_history=False,
):
    """Run an Optuna study for *model_type* and return the Study object."""

    training_scores.clear()
    val_scores.clear()

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(train_data, y_train, val_data, y_val, trial, model_type),
        n_trials=n_trials,
        show_progress_bar=True,
        )
    # ── Print summary ─────────────────────────────────────────────────────────
    best_trial = study.best_trial
    print(f"  Finished trials : {len(study.trials)}")
    print(f"  Best CV F1      : {best_trial.value:.4f}")
    print(f"  Best train F1   : {training_scores[best_trial.number]:.4f}")
    print(f"  Best params     : {study.best_params}")

    # Final score on the holdout validation set
    best_model = build_model(model_type, study.best_params)

    if model_type == "GradientBoosting":
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        sample_weights = np.where(y_train == 1, neg / pos, 1.0).ravel()  # ← .ravel() here too
        best_model.fit(train_data, y_train, sample_weight=sample_weights)
    else:
        best_model.fit(train_data, y_train)

    final_preds = best_model.predict(val_data)
    print(f"  Holdout val F1  : {f1_score(y_val, final_preds):.4f}")

    # ── Convergence plot ──────────────────────────────────────────────────────
    plt.figure()
    plt.plot(training_scores, label="Training F1")
    plt.plot(val_scores,      label="CV Validation F1")
    plt.xlabel("Trial")
    plt.ylabel("F1 Score")
    plt.title(f"{model_type} — Optuna Convergence")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if hyper_importance:
        display(optuna.visualization.plot_param_importances(study))
    if slice_plot:
        display(optuna.visualization.plot_slice(study, params=list(study.best_params.keys())))
    if optimization_history:
        display(optuna.visualization.plot_optimization_history(study))

    return study





# ── Train all models and collect results ──────────────────────────────────────

def train_all_models(models, train_data, y_train, val_data, y_val, test_data=None, n_trials=100, **kwargs):
    """
    Optimise every model in *models*, evaluate on the holdout val set,
    and optionally predict on *test_data*.

    Parameters
    ----------
    models    : dict  {model_name: optimizer_fn}
    test_data : array-like or DataFrame, optional
                If provided, predictions are added to the results DataFrame.

    Returns
    -------
    pd.DataFrame  sorted by val F1 descending; one row per model.
    """
    results = []

    for model_name, optimizer_fn in models.items():
        print(f"\n{'=' * 55}")
        print(f"  Optimising {model_name} …")
        print(f"{'=' * 55}")

        study = optimizer_fn(train_data, y_train, val_data, y_val, n_trials=n_trials, **kwargs)

        # Re-fit best model on full training data
        # Re-fit best model on full training data
    best_model = build_model(model_name, study.best_params)

    if model_name == "GradientBoosting":
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        sample_weights = np.where(y_train == 1, neg / pos, 1.0).ravel()
        best_model.fit(train_data, y_train, sample_weight=sample_weights)
    else:
        best_model.fit(train_data, y_train)
        best_model.fit(train_data, y_train)

        # Holdout validation metrics
        val_preds = best_model.predict(val_data)
        row = {
            "MODEL":      model_name,
            "PARAMETERS": study.best_params,
            "F1":         round(f1_score(y_val, val_preds),        4),
            "ACCURACY":   round(accuracy_score(y_val, val_preds),  4),
            "PRECISION":  round(precision_score(y_val, val_preds), 4),
            "RECALL":     round(recall_score(y_val, val_preds),    4),
        }

        # Optional test predictions
        if test_data is not None:
            row["TEST_PREDICTIONS"] = best_model.predict(test_data).tolist()

        results.append(row)
        print(f"  ✓ {model_name} — val F1: {row['F1']}")

    results_df = (
        pd.DataFrame(results)
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
    )
    return results_df