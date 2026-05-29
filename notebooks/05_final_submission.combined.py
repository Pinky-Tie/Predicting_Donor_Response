"""Importing the libraries needed for data handling, preprocessing, modeling, and evaluation"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


"""Adding the project root to the Python path so notebook imports work correctly"""
PROJECT_ROOT = Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)



"""Loading the features datasets"""
train_data = pd.read_csv(PROJECT_ROOT / "project_data" / "X_train_clean.csv", index_col=0)
val_data = pd.read_csv(PROJECT_ROOT / "project_data" / "X_val_clean.csv", index_col=0)
test_data = pd.read_csv(PROJECT_ROOT / "project_data" / "X_test_clean.csv", index_col=0)

'''Loading the y datasets'''
y_train = pd.read_csv(PROJECT_ROOT / "project_data" / "y_train_clean.csv", index_col=0).squeeze()
y_val   = pd.read_csv(PROJECT_ROOT / "project_data" / "y_val_clean.csv",   index_col=0).squeeze()



# Define the Model variable by evaluating the raw_params string





Model = DecisionTreeClassifier(max_depth=3, min_samples_split=9, min_samples_leaf=26, max_features='sqrt')
Model




full_X = np.concatenate([train_data, val_data])
full_y = np.concatenate([y_train, y_val]).ravel()

# Sample weights for class imbalance (GBM has no class_weight parameter)
neg = (full_y == 0).sum()
pos = (full_y == 1).sum()
sample_weights_full = np.where(full_y == 1, neg / pos, 1.0).ravel()

# Cross-validation with sample weights passed through each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
fold_scores = []
for train_idx, val_idx in cv.split(full_X, full_y):
    X_tr      = full_X[train_idx]
    X_val_cv  = full_X[val_idx]
    y_tr      = full_y[train_idx]
    y_val_cv  = full_y[val_idx]
    w_tr      = sample_weights_full[train_idx]
    Model.fit(X_tr, y_tr, sample_weight=w_tr)
    fold_scores.append(f1_score(y_val_cv, Model.predict(X_val_cv)))

print(f"CV F1 scores: {fold_scores}")
print(f"CV F1 mean: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores)*2:.4f})")

# Final fit on full data with sample weights
Model.fit(full_X, full_y, sample_weight=sample_weights_full)

# Final evaluation on held-out validation set (informative only — val is now in training)
val_f1 = f1_score(y_val, Model.predict(val_data))
print(f"Final validation F1: {val_f1:.4f}")

# Align test_data columns with train_data columns and predict
test_data_aligned = test_data[train_data.columns]
predModel = Model.predict(test_data_aligned)
Model_predictions = pd.DataFrame(predModel, index=test_data_aligned.index, columns=["TARGET_B"])
Model_predictions

Model_predictions.to_csv((PROJECT_ROOT / "project_data" / "Predictions.csv"))