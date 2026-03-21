"""Feature engineering focused on donor response behaviour."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class DonorFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create simple donor-specific features before model training."""
    def __init__(self, copy: bool = True):
        self.copy = copy
    def fit(self, X: pd.DataFrame, y=None):
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "DonorFeatureEngineer expects a pandas DataFrame so it can preserve column names."
            )
        df = X.copy() if self.copy else X

        self._add_ratio_feature(
            df,
            numerator="LIFETIME_GIFT_AMOUNT",
            denominator="LIFETIME_GIFT_COUNT",
            new_column="AVG_GIFT_PER_DONATION",
        )
        self._add_ratio_feature(
            df,
            numerator="LIFETIME_GIFT_AMOUNT",
            denominator="LIFETIME_PROM",
            new_column="GIFT_AMOUNT_PER_PROMOTION",
        )
        self._add_difference_feature(
            df,
            left="MONTHS_SINCE_FIRST_GIFT",
            right="MONTHS_SINCE_LAST_GIFT",
            new_column="MONTHS_BETWEEN_FIRST_AND_LAST_GIFT",
        )
        self._add_difference_feature(
            df,
            left="RECENT_RESPONSE_PROP",
            right="RECENT_CARD_RESPONSE_PROP",
            new_column="RECENT_RESPONSE_RATIO_GAP",
        )
        return df
    @staticmethod
    def _add_ratio_feature(
        df: pd.DataFrame,
        numerator: str,
        denominator: str,
        new_column: str,
    ) -> None:
        if numerator not in df.columns or denominator not in df.columns:
            return
        denominator_values = df[denominator].replace(0, np.nan)
        df[new_column] = df[numerator] / denominator_values
    @staticmethod
    def _add_difference_feature(
        df: pd.DataFrame,
        left: str,
        right: str,
        new_column: str,
    ) -> None:
        if left not in df.columns or right not in df.columns:
            return

        df[new_column] = df[left] - df[right]
