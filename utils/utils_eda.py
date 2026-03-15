#data handling 
import numpy as np
import pandas as pd

#display and visualization
import plotly.express as px
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
#machine learning and preprocessing
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#warning handling
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def analyze_missingness_patterns(df, cols_missing):
    """
    Analyze and visualize missingness patterns among specified columns.

    Parameters
    ----------
    df : pandas.DataFrame
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
    miss = df[cols_missing].isna()

    # Pairwise correlation of missingness
    miss_corr = miss.astype(float).corr()

    print("Pairwise Missingness Correlation Matrix:")
    display(miss_corr.style.background_gradient(cmap='RdYlGn_r').format("{:.2f}"))

    # Pairwise overlap counts
    pair_counts = pd.DataFrame(
        {(c1, c2): (df[c1].isna() & df[c2].isna()).sum()
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


def reduce_memory_usage(df):
    for col in df.columns:
        if df[col].dtype != 'object':
            if 'int' in str(df[col].dtype):
                df[col] = df[col].astype('int32')
            elif 'float' in str(df[col].dtype):
                df[col] = df[col].astype('float32')
    return df