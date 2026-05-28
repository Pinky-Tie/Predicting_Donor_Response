import numpy as np
import pandas as pd

def force_incoherence_to_null(df, rules=None):
    """
    Apply domain-specific validation rules to mark incoherent values as missing,
    and attempt to coerce a set of numeric-like columns to integer types.

    The function performs two steps:
    1. Apply boolean rules per-column to identify incoherent entries and set them to
       `np.nan` to indicate missingness.
    2. For a small set of integer-like columns, try to coerce non-null values to
       numeric, round them, and convert to pandas' nullable integer dtype (`Int64`),
       preserving `NaN` where coercion fails or values were masked.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to clean in-place (the object is mutated and returned).
    rules : dict[str, callable], optional
        Mapping of column name to a callable that accepts a Series and returns a
        boolean mask indicating invalid rows. If None, a project-default rule
        set is used (e.g. negative ages, children < 0, TARGET_B not in {0,1},
        proportions outside [0,1], etc.).

    Returns
    -------
    pd.DataFrame
        The same dataframe reference with incoherent values set to `NaN` and
        selected columns coerced towards integer type where possible.
    """
    if rules is None:
        rules = {
            "DONOR_AGE": lambda s: (s < 0) | (s > 100),
            "CHILDREN": lambda s: s < 0,
            "TARGET_B": lambda s: ~s.isin([0, 1]),
            "RECENT_RESPONSE_PROP": lambda s: (s < 0) | (s > 1),
            "RECENT_CARD_RESPONSE_PROP": lambda s: (s < 0) | (s > 1),
            "RECENT_AVG_CARD_GIFT_AMT": lambda s: s < 0,
            "MEDIAN_HOUSEHOLD_INCOME": lambda s: s < 0,
            "PER_CAPITA_INCOME": lambda s: s < 0,
        }

    for col, condition in rules.items():
        if col not in df.columns:
            continue
        mask = condition(df[col])
        mask = mask.fillna(False)
        df.loc[mask, col] = np.nan
    # After masking out-of-range / incoherent values, attempt to coerce
    # certain numeric columns to integers when possible by rounding.
    # Keep NaNs introduced by the masking above.
    int_columns = [
        "DONOR_AGE",
        "CHILDREN",
        "TARGET_B",
        "RECENT_AVG_CARD_GIFT_AMT",
        "MEDIAN_HOUSEHOLD_INCOME",
        "PER_CAPITA_INCOME",
    ]

    for col in int_columns:
        if col not in df.columns:
            continue

        # Only operate on non-null values (we want to preserve NaNs)
        non_null_mask = df[col].notna()
        if not non_null_mask.any():
            continue

        # Try to coerce to numeric; non-numeric values become NaN and are left unchanged
        coerced = pd.to_numeric(df.loc[non_null_mask, col], errors="coerce")
        # Round numeric values and convert to nullable integer dtype
        coerced = coerced.round()

        # Assign back only where coercion produced a numeric value
        valid_coercion = coerced.notna()
        if valid_coercion.any():
            # Use pandas nullable integer type to preserve missing values
            df.loc[non_null_mask, col] = df.loc[non_null_mask, col].where(
                ~valid_coercion, coerced.astype("Int64")
            )

    return df
