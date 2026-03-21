"""Submission builders for the donor response competition format."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .data_loading import DEFAULT_ID_COLUMN, DEFAULT_TARGET_COLUMN
def build_submission(
    test_ids,
    predictions,
    id_column: str = DEFAULT_ID_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    """Build a competition-style submission dataframe."""
    submission = pd.DataFrame(
        {
            id_column: pd.Series(test_ids).values,
            target_column: pd.Series(predictions).values,
        }
    )
    return submission
def save_submission(submission_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist a submission dataframe to disk and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return output_path
