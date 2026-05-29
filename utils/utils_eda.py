#data handling 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#display and visualization
from IPython.display import display
#machine learning and preprocessing
#warning handling
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def analyse_mnar(data: pd.DataFrame, target: str = "TARGET_B", id_col: str = "CONTROL_NUMBER") -> pd.DataFrame:
    """
    Test each column with missing values for MNAR behaviour.

    For each column, builds a binary missingness indicator and tests whether
    that indicator is associated with other observed variables using a
    logistic regression. A high AUC means missingness can be predicted from
    other columns — strong evidence of MNAR.

    Parameters
    ----------
    data : pd.DataFrame
    target : str
        Target column to exclude from predictors.
    id_col : str
        ID column to exclude from predictors.

    Returns
    -------
    pd.DataFrame
        One row per column with missing values, sorted by AUC descending.
        Columns: variable, pct_missing, auc, verdict, top_predictors.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    drop_cols  = {target, id_col}
    candidates = [c for c in data.columns if data[c].isna().any() and c not in drop_cols]
    predictors = [c for c in data.select_dtypes(include="number").columns if c not in drop_cols]

    records = []

    for col in candidates:

        # Build missingness indicator
        indicator = data[col].isna().astype(int)

        # Use all other numeric columns (that are not this col) as predictors
        feats = [c for c in predictors if c != col]
        X = data[feats].copy()

        # Drop rows where ALL predictors are also missing
        X = X.dropna(how="all")
        y = indicator.loc[X.index]

        # Fill remaining NaNs in predictors with median for the regression
        X = X.fillna(X.median())

        if y.nunique() < 2:
            continue  # can't fit if no variation

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        lr = LogisticRegression(max_iter=500, random_state=1)
        lr.fit(X_sc, y)
        auc = roc_auc_score(y, lr.predict_proba(X_sc)[:, 1])

        # Top 3 predictors of missingness by absolute coefficient
        coef_series = pd.Series(np.abs(lr.coef_[0]), index=feats).sort_values(ascending=False)
        top3 = ", ".join(coef_series.head(3).index.tolist())

        if auc >= 0.70:
            verdict = "MNAR"
        elif auc >= 0.60:
            verdict = "Likely MAR"
        else:
            verdict = "MCAR / weak"

        records.append({
            "variable":       col,
            "pct_missing":    round(data[col].isna().mean() * 100, 2),
            "auc":            round(auc, 4),
            "verdict":        verdict,
            "top_predictors": top3,
        })

    results = (
        pd.DataFrame(records)
          .sort_values("auc", ascending=False)
          .reset_index(drop=True)
    )
    return results

def explain_mnar(data: pd.DataFrame) -> None:
    """
    Plot a focused diagnostic chart for each confirmed MNAR variable.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # WEALTH_RATING — missing rate by lifetime gift quintile
    (
        data.assign(quintile=pd.qcut(data["LIFETIME_GIFT_AMOUNT"], q=5, duplicates="drop"))
          .groupby("quintile", observed=True)["WEALTH_RATING"]
          .apply(lambda x: x.isna().mean())
          .plot(kind="bar", ax=axes[0], color="#C0392B", alpha=0.85)
    )
    axes[0].set_title("WEALTH_RATING\nmissing rate by lifetime gift quintile")
    axes[0].set_xlabel("Lifetime gift amount (low to high)")
    axes[0].set_ylabel("Missing rate")
    axes[0].tick_params(axis="x", rotation=30)

    # DONOR_AGE — missing rate by home ownership
    (
        data.groupby("HOME_OWNER", observed=True)["DONOR_AGE"]
          .apply(lambda x: x.isna().mean())
          .rename(index={"H": "Homeowner", "U": "Renter"})
          .plot(kind="bar", ax=axes[1], color=["#2E86C1", "#C0392B"], alpha=0.85)
    )
    axes[1].set_title("DONOR_AGE\nmissing rate by tenure type")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Missing rate")
    axes[1].tick_params(axis="x", rotation=0)

    # MONTHS_SINCE_LAST_PROM_RESP — missing rate by recency status
    (
        data.groupby("RECENCY_STATUS_96NK", observed=True)["MONTHS_SINCE_LAST_PROM_RESP"]
          .apply(lambda x: x.isna().mean())
          .sort_values(ascending=False)
          .plot(kind="bar", ax=axes[2], color="#8E44AD", alpha=0.85)
    )
    axes[2].set_title("MONTHS_SINCE_LAST_PROM_RESP\nmissing rate by recency status")
    axes[2].set_xlabel("Recency status")
    axes[2].set_ylabel("Missing rate")
    axes[2].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.show()


def analyze_missingness_patterns(data, cols_missing):
    """
    Analyze and visualize missingness patterns among specified columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to analyze.
    cols_missing : list of str
        List of column names to check for missingness correlations.

    Returns
    -------
    dict
        Summary dictionary with:
        - 'corr_matrix': DataFrame of missingness correlations
        - 'pair_missing_counts': DataFrame of pairwise missing overlaps
        - 'rows_all_missing': Number of rows where key columns are all missing
        - 'missing_indices': Index of rows where all key columns are missing
    """

    # Boolean mask of missingness for the selected columns
    miss = data[cols_missing].isna()

    # Pairwise correlation of missingness
    miss_corr = miss.astype(float).corr()

    print("Pairwise Missingness Correlation Matrix:")
    display(miss_corr.style.background_gradient(cmap='RdYlGn_r').format("{:.2f}"))

    # Pairwise overlap counts
    pair_counts = pd.DataFrame(
        {(c1, c2): (data[c1].isna() & data[c2].isna()).sum()
         for i, c1 in enumerate(cols_missing) for c2 in cols_missing[i+1:]},
        index=['count']
    ).T.sort_values('count', ascending=False)

    print("\n Top 10 Pairs with Highest Overlapping Missing Values:")
    display(pair_counts.head(10))

    # Return a summary dictionary for further use
    return {
        'corr_matrix': miss_corr,
        'pair_missing_counts': pair_counts
    }


def reduce_memory_usage(data):
    for col in data.columns:
        if data[col].dtype != 'object':
            if 'int' in str(data[col].dtype):
                data[col] = data[col].astype('int32')
            elif 'float' in str(data[col].dtype):
                data[col] = data[col].astype('float32')
    return data