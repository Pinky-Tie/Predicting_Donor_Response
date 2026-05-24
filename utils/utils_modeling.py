"""Importing the libraries needed for data handling, preprocessing, modeling, and evaluation"""

from matplotlib import pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from IPython.display import display
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
 
training_scores = []
val_scores = []



def objective(train_data, y_train, val_data, y_val, trial, model_type):
    if model_type == 'GradientBoosting':
        # Define hyperparameters to be optimized for GradientBoosting
        n_estimators = trial.suggest_int("n_estimators", 2, 500)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 1.0, log=True)
        subsample = trial.suggest_float("subsample", 0.2, 1.0)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 6)
        max_features = trial.suggest_categorical("max_features", [2, 0.5, 'sqrt', 'log2', None])
        max_depth = trial.suggest_int("max_depth", 2, 6)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=1,
        )

    elif model_type == 'LogisticRegression':
        C = trial.suggest_float('C', 0.001, 200)
        solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])

        model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            fit_intercept=fit_intercept,
            random_state=1,
        )

    elif model_type == 'AdaBoost':
        estimator = DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 1, 6),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 8),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            random_state=1,
        )

        n_estimators = trial.suggest_int("n_estimators", 2, 500)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 1.0, log=True)
        algorithm = trial.suggest_categorical('algorithm', ["SAMME", "SAMME.R"])

        model = AdaBoostClassifier(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=14,
        )

    elif model_type == 'DecisionTree':
        max_depth = trial.suggest_int("max_depth", 3, 7)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 15)

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=1,
        )

    elif model_type == 'RandomForestClassifier':
        n_estimators = trial.suggest_int('n_estimators', 7, 25)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        max_depth = trial.suggest_int('max_depth', 10, 14)
        min_samples_split = trial.suggest_int('min_samples_split', 5, 14)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 11)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        warm_start = trial.suggest_categorical('warm_start', [True, False])
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 120)
        min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=1,
            max_leaf_nodes=max_leaf_nodes,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            bootstrap=bootstrap,
            warm_start=warm_start,
            max_features=max_features,
        )

    elif model_type == 'GaussianNB':
        var_smoothing = trial.suggest_int('var_smoothing', -12, -6)
        model = GaussianNB(var_smoothing=10 ** var_smoothing)

    elif model_type == 'KNeighborsClassifier':
        n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        leaf_size = trial.suggest_int('leaf_size', 10, 100)
        p = trial.suggest_int('p', 1, 6)

        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
        )

    elif model_type == 'MLPClassifier':
        activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
        solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
        shuffle = trial.suggest_categorical('shuffle', [True, False])
        early_stopping = trial.suggest_categorical('early_stopping', [True, False])
        max_iter = trial.suggest_int('max_iter', 100, 1000)

        model = MLPClassifier(
            activation=activation,
            solver=solver,
            learning_rate=learning_rate,
            shuffle=shuffle,
            early_stopping=early_stopping,
            max_iter=max_iter,
        )

    elif model_type == 'Stacking':
        rf_params = {
            'n_estimators': 16,
            'criterion': 'log_loss',
            'max_depth': 10,
            'min_samples_split': 7,
            'min_samples_leaf': 6,
            'bootstrap': False,
            'warm_start': False,
        }
        DTC_params = {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 11}
        lr_params = {'C': 13.201220684960449, 'solver': 'newton-cg', 'max_iter': 419, 'fit_intercept': True}
        KNN_params = {'n_neighbors': 27, 'weights': 'distance', 'algorithm': 'ball_tree', 'leaf_size': 55, 'p': 2}

        meta_model_choice = trial.suggest_categorical('final_estimator', ['KNN', 'LogisticRegression'])
        if meta_model_choice == 'LogisticRegression':
            meta_model = LogisticRegression(**lr_params)
        else:
            meta_model = KNeighborsClassifier(**KNN_params)

        model = StackingClassifier(
            estimators=[
                ('LogisticRegression', LogisticRegression(**lr_params)),
                ('DecisionTreeClassifier', DecisionTreeClassifier(**DTC_params)),
                ('RandomF', RandomForestClassifier(**rf_params)),
            ],
            final_estimator=meta_model,
        )

    else:
        raise ValueError("Invalid model_type.")

    # Cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    cv_scores = cross_val_score(model, train_data, y_train, cv=cv, scoring='f1')
    validation_score = cv_scores.mean()

    # Train on full train_data to get the training score
    model.fit(train_data, y_train)
    train_preds = model.predict(train_data)
    training_score = f1_score(y_train, train_preds)

    training_scores.append(training_score)
    val_scores.append(validation_score)

    return validation_score  # Optuna maximizes this



def optimize_with_optuna(model_type, train_data, y_train, val_data, y_val,
                         n_trials=100, hyper_importance=False, 
                         slice_plot=False, optimization_history=False):
    
    training_scores.clear()
    val_scores.clear()


    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, model_type, train_data, y_train, val_data, y_val),
        n_trials=n_trials
    )

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  CV Validation Score (F1): ", trial.value)
    print("  Train Score (F1): ", training_scores[trial.number])
    print("  Params: ", study.best_params)

    # Final evaluation on holdout val_data
    best_model = build_model(model_type, study.best_params)
    best_model.fit(train_data, y_train)
    final_preds = best_model.predict(val_data)
    final_score = f1_score(y_val, final_preds)
    print(f"  Final Val Score on holdout (F1): {final_score:.4f}")

    # Plots
    plt.plot(training_scores, label='Training Score (F1)')
    plt.plot(val_scores, label='Validation Score (F1)')
    plt.xlabel('Trials')
    plt.ylabel('F1 Score')
    plt.title(f'{model_type} - Optuna Trials')
    plt.legend()
    plt.show()

    if hyper_importance:
        display(optuna.visualization.plot_param_importances(study))
    if slice_plot:
        params_to_plot = study.best_params.keys()
        display(optuna.visualization.plot_slice(study, params=params_to_plot))
    if optimization_history:
        display(optuna.visualization.plot_optimization_history(study))

    return study


def build_model(model_type, params):
    if model_type == 'GradientBoosting':
        return GradientBoostingClassifier(**params, random_state=1)
    elif model_type == 'LogisticRegression':
        return LogisticRegression(**params, random_state=1)
    elif model_type == 'AdaBoost':
        return AdaBoostClassifier(**params, random_state=1)
    elif model_type == 'DecisionTree':
        return DecisionTreeClassifier(**params, random_state=1)
    elif model_type == 'RandomForestClassifier':
        return RandomForestClassifier(**params, random_state=1)
    elif model_type == 'GaussianNB':
        return GaussianNB(**params)
    elif model_type == 'KNeighborsClassifier':
        return KNeighborsClassifier(**params)
    elif model_type == 'MLPClassifier':
        return MLPClassifier(**params)
    



def train_all_models(models, train_data, y_train, val_data, y_val, test_data, n_trials=100, **kwargs):
    results = []

    for model_name, optimizer_fn in models.items():
        print(f"\n{'='*50}")
        print(f"Optimizing {model_name}...")
        print(f"{'='*50}")

        # Run Optuna optimization
        study = optimizer_fn(train_data, y_train, val_data, y_val, n_trials=n_trials, **kwargs)

        # Rebuild best model with best params
        best_model = build_model(model_name, study.best_params)
        best_model.fit(train_data, y_train)

        # Evaluate on val_data (holdout)
        val_preds = best_model.predict(val_data)
        results.append({
            "MODEL":      model_name,
            "PARAMETERS": study.best_params,
            "F1 SCORE":   round(f1_score(y_val, val_preds), 4),
            "ACCURACY":   round(accuracy_score(y_val, val_preds), 4),
            "PRECISION":  round(precision_score(y_val, val_preds), 4),
            "RECALL":     round(recall_score(y_val, val_preds), 4),
        })

        print(f" {model_name} done — F1: {results[-1]['F1 SCORE']}")

    results_df = pd.DataFrame(results).sort_values("F1 SCORE", ascending=False).reset_index(drop=True)
    return results_df

