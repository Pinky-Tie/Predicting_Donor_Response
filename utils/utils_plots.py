"""Plot helpers used across notebooks and model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

PAPER_BG_COLOR = "#FFFFFF"
PLOT_BG_COLOR = "#E5ECF6"
GRID_COLOR = "#FFFFFF"
FONT_COLOR = "#2A3F5F"
PRIMARY_COLOR = "#8C103B"
SECONDARY_COLOR = "#F97316"
TERTIARY_COLOR = "#F9C74F"
QUATERNARY_COLOR = "#90BE6D"
SUPPORT_COLOR = "#4D908E"

DONOR_DISCRETE_SEQUENCE = [
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    TERTIARY_COLOR,
    QUATERNARY_COLOR,
    SUPPORT_COLOR,
    "#577590",
    "#B56576",
    "#6D597A",
]

DONOR_SEQUENTIAL_SCALE = [
    (0.0, "#FFF4C2"),
    (0.2, "#FED976"),
    (0.45, "#F8961E"),
    (0.7, "#F94144"),
    (1.0, PRIMARY_COLOR),
]

DONOR_DIVERGING_SCALE = [
    (0.0, "#0B6E4F"),
    (0.25, QUATERNARY_COLOR),
    (0.5, "#FFF4C2"),
    (0.75, "#F8961E"),
    (1.0, PRIMARY_COLOR),
]

DONOR_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "donor_sequential",
    [color for _, color in DONOR_SEQUENTIAL_SCALE],
)


def _apply_plotly_theme(fig):
    """Apply the shared light blue dashboard theme to Plotly figures."""
    fig.update_layout(
        paper_bgcolor=PAPER_BG_COLOR,
        plot_bgcolor=PLOT_BG_COLOR,
        font={"color": FONT_COLOR},
        title={"font": {"color": FONT_COLOR, "size": 18}},
        legend={
            "bgcolor": "rgba(255,255,255,0.75)",
            "bordercolor": "rgba(0,0,0,0)",
            "font": {"color": FONT_COLOR},
        },
        coloraxis_colorbar={
            "title_font": {"color": FONT_COLOR},
            "tickfont": {"color": FONT_COLOR},
        },
        margin={"l": 80, "r": 40, "t": 80, "b": 60},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        tickfont={"color": FONT_COLOR},
        title_font={"color": FONT_COLOR},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        tickfont={"color": FONT_COLOR},
        title_font={"color": FONT_COLOR},
    )
    fig.for_each_annotation(lambda annotation: annotation.update(font={"color": FONT_COLOR}))
    return fig


def _apply_matplotlib_theme(ax, *, grid_axis: str | None = None):
    """Apply the shared theme to matplotlib axes."""
    ax.figure.patch.set_facecolor(PAPER_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)
    ax.title.set_color(FONT_COLOR)
    ax.xaxis.label.set_color(FONT_COLOR)
    ax.yaxis.label.set_color(FONT_COLOR)
    ax.tick_params(colors=FONT_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=1.1)
        ax.set_axisbelow(True)
    return ax


def _get_numeric_columns(df: pd.DataFrame, columns=None) -> list[str]:
    """Selecting numeric columns from the full dataset or from a provided subset."""
    if columns is None:
        return df.select_dtypes(include=np.number).columns.tolist()
    return [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


def plot_invalid_value_counts(
    df: pd.DataFrame,
    rules: dict[str, callable],
    title: str = "Invalid Value Counts by Feature",
):
    """Counting invalid values per feature based on user-defined validation rules."""
    invalid_counts = {}

    for column, rule in rules.items():
        if column not in df.columns:
            invalid_counts[column] = np.nan
            continue

        mask = rule(df[column])
        if isinstance(mask, (pd.Series, np.ndarray, list)):
            invalid_counts[column] = int(np.asarray(mask, dtype=bool).sum())
        else:
            raise ValueError(f"Rule for column '{column}' must return a boolean mask.")

    invalid_df = (
        pd.Series(invalid_counts, name="invalid_count")
        .dropna()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    fig = px.bar(
        invalid_df,
        x="invalid_count",
        y="feature",
        orientation="h",
        title=title,
        labels={"invalid_count": "Invalid Count", "feature": "Feature"},
        color="invalid_count",
        color_continuous_scale=DONOR_SEQUENTIAL_SCALE,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return _apply_plotly_theme(fig)


def plot_boxplots_numeric(
    df: pd.DataFrame,
    columns=None,
    title: str = "Boxplots for Numeric Features",
):
    """Displaying numeric feature distributions to help detect suspicious ranges and outliers."""
    numeric_columns = _get_numeric_columns(df, columns)
    if not numeric_columns:
        raise ValueError("No numeric columns available for boxplot visualization.")

    melted = df[numeric_columns].melt(var_name="feature", value_name="value")
    fig = px.box(
        melted,
        x="feature",
        y="value",
        color="feature",
        points="outliers",
        title=title,
        color_discrete_sequence=DONOR_DISCRETE_SEQUENCE,
    )
    fig.update_layout(showlegend=False, xaxis_title="Feature", yaxis_title="Value")
    return _apply_plotly_theme(fig)


def plot_outlier_boxplots(
    df: pd.DataFrame,
    columns=None,
    title: str = "Outlier Analysis with Boxplots",
):
    """Visualizing numeric distributions with boxplots to inspect potential outliers."""
    return plot_boxplots_numeric(df=df, columns=columns, title=title)


def plot_outlier_histograms(
    df: pd.DataFrame,
    columns=None,
    nbins: int = 30,
    facet_col_wrap: int = 2,
    log_x: bool = False,
    clip_percentiles: tuple[float, float] | None = (0.01, 0.99),
    histnorm: str | None = None,
    title: str = "Histograms for Numeric Features",
):
    """Displaying numeric feature distributions to inspect skewness and extreme values."""
    numeric_columns = _get_numeric_columns(df, columns)
    if not numeric_columns:
        raise ValueError("No numeric columns available for histogram visualization.")

    frames = []
    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty:
            continue

        if clip_percentiles is not None:
            lower_q, upper_q = clip_percentiles
            lower = series.quantile(lower_q)
            upper = series.quantile(upper_q)
            series = series[(series >= lower) & (series <= upper)]

        frames.append(pd.DataFrame({"feature": column, "value": series}))

    if not frames:
        raise ValueError("No numeric values available for histogram visualization after filtering.")

    melted = pd.concat(frames, ignore_index=True)
    fig = px.histogram(
        melted,
        x="value",
        facet_col="feature",
        facet_col_wrap=facet_col_wrap,
        nbins=nbins,
        title=title,
        log_x=log_x,
        histnorm=histnorm,
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(
        showlegend=False,
        width=max(1200, facet_col_wrap * 500),
        height=max(700, ((len(numeric_columns) + facet_col_wrap - 1) // facet_col_wrap) * 350),
    )
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return _apply_plotly_theme(fig)


def plot_outlier_counts_iqr(
    df: pd.DataFrame,
    columns=None,
    iqr_factor: float = 1.5,
    title: str = "Outlier Counts by Feature (IQR Rule)",
):
    """Counting potential outliers per numeric feature using the IQR rule."""
    numeric_columns = _get_numeric_columns(df, columns)
    if not numeric_columns:
        raise ValueError("No numeric columns available for outlier counting.")

    counts = {}
    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty:
            counts[column] = 0
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        counts[column] = int(((series < lower) | (series > upper)).sum())

    counts_df = (
        pd.Series(counts, name="outlier_count")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    fig = px.bar(
        counts_df,
        x="outlier_count",
        y="feature",
        orientation="h",
        title=title,
        labels={"outlier_count": "Potential Outlier Count", "feature": "Feature"},
        color="outlier_count",
        color_continuous_scale=DONOR_SEQUENTIAL_SCALE,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return _apply_plotly_theme(fig)


def plot_outlier_percentages_iqr(
    df: pd.DataFrame,
    columns=None,
    iqr_factor: float = 1.5,
    exclude_columns=None,
    title: str = "Outlier Percentage by Feature (IQR Rule)",
):
    """Computing the share of potential outliers per numeric feature using the IQR rule."""
    numeric_columns = _get_numeric_columns(df, columns)
    if exclude_columns is not None:
        excluded = set(exclude_columns)
        numeric_columns = [col for col in numeric_columns if col not in excluded]
    if not numeric_columns:
        raise ValueError("No numeric columns available for outlier percentage analysis.")

    percentages = {}
    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty:
            percentages[column] = 0.0
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        outlier_mask = (series < lower) | (series > upper)
        percentages[column] = float(outlier_mask.mean() * 100)

    percentages_df = (
        pd.Series(percentages, name="outlier_percent")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    fig = px.bar(
        percentages_df,
        x="outlier_percent",
        y="feature",
        orientation="h",
        title=title,
        labels={"outlier_percent": "Potential Outliers (%)", "feature": "Feature"},
        color="outlier_percent",
        color_continuous_scale=DONOR_SEQUENTIAL_SCALE,
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        width=1200,
        height=max(900, len(percentages_df) * 28),
    )
    return _apply_plotly_theme(fig)


def plot_categorical_frequencies(
    df: pd.DataFrame,
    columns=None,
    top_n: int | None = None,
    facet_col_wrap: int = 2,
    title: str = "Categorical Feature Frequencies",
):
    """Visualizing category frequencies for selected categorical variables."""
    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        columns = [col for col in columns if col in df.columns]

    if not columns:
        raise ValueError("No categorical columns available for frequency visualization.")

    frames = []
    for column in columns:
        counts = df[column].fillna("Missing").value_counts(dropna=False)
        if top_n is not None:
            counts = counts.head(top_n)
        freq_df = counts.reset_index()
        freq_df.columns = ["category", "count"]
        freq_df["feature"] = column
        frames.append(freq_df)

    plot_df = pd.concat(frames, ignore_index=True)

    fig = px.bar(
        plot_df,
        x="category",
        y="count",
        color="category",
        facet_col="feature",
        facet_col_wrap=facet_col_wrap,
        title=title,
        color_discrete_sequence=DONOR_DISCRETE_SEQUENCE,
    )
    n_rows = (len(columns) + facet_col_wrap - 1) // facet_col_wrap
    fig.update_layout(
        showlegend=False,
        width=max(1200, facet_col_wrap * 560),
        height=max(550, n_rows * 440),
        bargap=0.25,
    )
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return _apply_plotly_theme(fig)


def plot_categorical_target_rate(
    df: pd.DataFrame,
    column: str | list[str],
    target: str = "TARGET_B",
    facet_col_wrap: int = 2,
    title: str | None = None,
):
    """Comparing the target rate across categories of a categorical variable."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not present in the DataFrame.")

    if isinstance(column, str):
        columns = [column]
    else:
        columns = [col for col in column if col in df.columns]

    if not columns:
        raise ValueError("No valid categorical columns were provided.")

    frames = []
    for col in columns:
        summary_df = (
            df[[col, target]]
            .copy()
            .assign(**{col: lambda frame: frame[col].fillna("Missing")})
            .groupby(col, dropna=False)[target]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        summary_df.columns = ["category", "target_rate"]
        summary_df["feature"] = col
        frames.append(summary_df)

    plot_df = pd.concat(frames, ignore_index=True)

    if title is None:
        title = (
            f"Target Rate by Category for {columns[0]}"
            if len(columns) == 1
            else "Target Rate by Category Across Categorical Features"
        )

    fig = px.bar(
        plot_df,
        x="category",
        y="target_rate",
        color="category",
        facet_col="feature" if len(columns) > 1 else None,
        facet_col_wrap=facet_col_wrap if len(columns) > 1 else None,
        title=title,
        labels={"category": "Category", "target_rate": f"Mean {target}"},
        color_discrete_sequence=DONOR_DISCRETE_SEQUENCE,
    )

    if len(columns) > 1:
        n_rows = (len(columns) + facet_col_wrap - 1) // facet_col_wrap
        fig.update_layout(
            showlegend=False,
            width=max(1500, facet_col_wrap * 620),
            height=max(700, n_rows * 480),
            bargap=0.25,
        )
        fig.update_xaxes(matches=None, showticklabels=True)
        fig.update_yaxes(showticklabels=True)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    else:
        fig.update_layout(showlegend=False)

    return _apply_plotly_theme(fig)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    columns=None,
    mask_upper: bool = True,
    target: str | None = None,
    top_n: int | None = None,
    sort_by_target: bool = False,
    show_text: bool = False,
    title: str = "Feature Correlation Heatmap",
):
    """Plotting a correlation heatmap for numeric variables."""
    numeric_columns = _get_numeric_columns(df, columns)
    if not numeric_columns:
        raise ValueError("No numeric columns available for correlation analysis.")

    if target is not None and target not in numeric_columns:
        raise ValueError(f"Target column '{target}' must be numeric and present in the selected columns.")

    if target is not None and top_n is not None:
        ranked_columns = (
            df[numeric_columns]
            .corr(method=method)[target]
            .drop(labels=[target])
            .abs()
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )
        numeric_columns = ranked_columns + [target]
    elif top_n is not None:
        numeric_columns = numeric_columns[:top_n]

    corr = df[numeric_columns].corr(method=method)

    if sort_by_target and target is not None:
        ordered_columns = (
            corr[target]
            .drop(labels=[target])
            .abs()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        ordered_columns = [target] + ordered_columns
        corr = corr.loc[ordered_columns, ordered_columns]

    if mask_upper:
        corr = corr.mask(np.triu(np.ones_like(corr, dtype=bool), k=1))

    fig = px.imshow(
        corr,
        text_auto=".2f" if show_text else False,
        aspect="auto",
        color_continuous_scale=DONOR_DIVERGING_SCALE,
        zmin=-1,
        zmax=1,
        title=title,
    )
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Feature",
        width=max(1400, len(corr.columns) * 55),
        height=max(1200, len(corr.columns) * 55),
    )
    if show_text:
        fig.update_traces(textfont={"size": 10})
    return _apply_plotly_theme(fig)


def plot_missing_percentages(
    df: pd.DataFrame,
    top_n: int | None = None,
    title: str = "Missing Values Percentage by Feature",
):
    """Showing the percentage of missing values per feature."""
    missing_pct = ((df.isna().mean() * 100).sort_values(ascending=False)).reset_index()
    missing_pct.columns = ["feature", "missing_percent"]

    if top_n is not None:
        missing_pct = missing_pct.head(top_n)

    fig = px.bar(
        missing_pct,
        x="missing_percent",
        y="feature",
        orientation="h",
        title=title,
        labels={"missing_percent": "Missing (%)", "feature": "Feature"},
        color="missing_percent",
        color_continuous_scale=DONOR_SEQUENTIAL_SCALE,
    )
    fig.update_layout(
        yaxis={
            "categoryorder": "array",
            "categoryarray": missing_pct["feature"].iloc[::-1].tolist(),
            "tickmode": "array",
            "tickvals": missing_pct["feature"].tolist(),
            "ticktext": missing_pct["feature"].tolist(),
            "automargin": True,
        },
        height=max(700, 24 * len(missing_pct)),
    )
    return _apply_plotly_theme(fig)


def plot_missing_overlap_heatmap(
    df: pd.DataFrame,
    columns=None,
    top_n: int | None = None,
    exclude_columns=None,
    mask_upper: bool = False,
    title: str = "Missingness Correlation Heatmap",
):
    """Comparing missing-value patterns across features through indicator correlations."""
    selected_columns = [col for col in (columns or df.columns.tolist()) if col in df.columns]
    if exclude_columns is not None:
        excluded = set(exclude_columns)
        selected_columns = [col for col in selected_columns if col not in excluded]

    if top_n is not None:
        selected_columns = (
            df[selected_columns]
            .isna()
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )

    if not selected_columns:
        raise ValueError("No valid columns available for missing-value overlap analysis.")

    missing_matrix = df[selected_columns].isna().astype(int)
    corr = missing_matrix.corr()
    if mask_upper:
        corr = corr.mask(np.triu(np.ones_like(corr, dtype=bool), k=1))

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=DONOR_DIVERGING_SCALE,
        zmin=-1,
        zmax=1,
        title=title,
    )
    fig.update_layout(xaxis_title="Feature", yaxis_title="Feature")
    return _apply_plotly_theme(fig)


def plot_missing_vs_target(
    df: pd.DataFrame,
    target: str = "TARGET_B",
    columns=None,
    top_n: int = 15,
    title: str | None = None,
):
    """Comparing target rates between missing and non-missing groups for selected features."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not present in the DataFrame.")

    selected_columns = [col for col in (columns or df.columns.tolist()) if col in df.columns and col != target]
    selected_columns = [col for col in selected_columns if df[col].isna().any()]
    if not selected_columns:
        raise ValueError("No columns with missing values available for target comparison.")

    ranked_columns = (
        df[selected_columns]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )

    rows = []
    for column in ranked_columns:
        missing_mask = df[column].isna()
        rows.append(
            {
                "feature": column,
                "group": "Missing",
                "target_rate": df.loc[missing_mask, target].mean(),
            }
        )
        rows.append(
            {
                "feature": column,
                "group": "Not Missing",
                "target_rate": df.loc[~missing_mask, target].mean(),
            }
        )

    summary_df = pd.DataFrame(rows)

    if title is None:
        title = f"Target Rate by Missingness Status for Top {len(ranked_columns)} Features"

    fig = px.bar(
        summary_df,
        x="target_rate",
        y="feature",
        color="group",
        barmode="group",
        orientation="h",
        title=title,
        labels={"target_rate": f"Mean {target}", "feature": "Feature", "group": "Group"},
        color_discrete_map={"Missing": PRIMARY_COLOR, "Not Missing": TERTIARY_COLOR},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return _apply_plotly_theme(fig)


def plot_target_correlation(
    df: pd.DataFrame,
    target: str = "TARGET_B",
    method: str = "pearson",
    columns=None,
    top_n: int = 15,
    title: str | None = None,
):
    """Ranking numeric features by their correlation with the target variable."""
    numeric_columns = _get_numeric_columns(df, columns)
    if target not in numeric_columns:
        raise ValueError(f"Target column '{target}' must be numeric and present in the selected columns.")

    corr_with_target = (
        df[numeric_columns]
        .corr(method=method)[target]
        .drop(labels=[target])
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(top_n)
        .sort_values()
        .reset_index()
    )
    corr_with_target.columns = ["feature", "correlation"]

    if title is None:
        title = f"Top {min(top_n, len(corr_with_target))} Feature Correlations with {target}"

    fig = px.bar(
        corr_with_target,
        x="correlation",
        y="feature",
        orientation="h",
        title=title,
        labels={"correlation": f"Correlation with {target}", "feature": "Feature"},
        color="correlation",
        color_continuous_scale=DONOR_DIVERGING_SCALE,
        range_color=[-1, 1],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.add_vline(x=0, line_dash="dash", line_color=FONT_COLOR)
    return _apply_plotly_theme(fig)


def plot_confusion_matrix_heatmap(
    y_true,
    y_pred,
    labels: tuple[str, str] = ("No Donation", "Donation"),
    ax=None,
):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=DONOR_SEQUENTIAL_CMAP,
        cbar=False,
        linewidths=1.0,
        linecolor=GRID_COLOR,
        ax=ax,
    )
    threshold = cm.max() / 2 if cm.size else 0
    for text, value in zip(ax.texts, cm.flatten()):
        text.set_color("white" if value > threshold else FONT_COLOR)
        text.set_fontweight("bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title("Confusion Matrix")
    return _apply_matplotlib_theme(ax)


def plot_probability_distribution(y_proba, ax=None, bins: int = 20):
    """Plot the distribution of positive-class predicted probabilities."""
    scores = np.asarray(y_proba)
    if scores.ndim == 2 and scores.shape[1] >= 2:
        scores = scores[:, 1]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.hist(scores, bins=bins, color=PRIMARY_COLOR, alpha=0.88, edgecolor="white")
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Positive Class Probability")
    ax.set_ylabel("Count")
    return _apply_matplotlib_theme(ax, grid_axis="y")
