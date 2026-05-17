import pandas as pd
from logging import Logger
from typing import Optional

# Import functions from the preprocessing_pipeline
from .preprocessing_pipeline.preprocessing_cleaning_invalid import *
from .preprocessing_pipeline.preprocessing_encoding import *
from .preprocessing_pipeline.preprocessing_feature_engineering import *
from .preprocessing_pipeline.preprocessing_missing_values import *
from .preprocessing_pipeline.preprocessing_scaling import *
from .preprocessing_pipeline.preproccessing_outliers import *


def log_preprocessing_step(step: str, logger: Optional[Logger] = None) -> None:
    message = f"[preprocess_data] {step}"
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def preprocess_data(
    data_original: pd.DataFrame,
    outlier_columns=None,
    outlier_method="rescale",
    IQR_value=1.5,
    encode=None,
    logger: Optional[Logger] = None,
    cols_to_drop: list[str] = None
) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values, encoding categorical variables,
    and performing feature engineering.
    Runs each of the preprocessing steps in sequence, ensuring that the data is clean and ready for modeling.
    Feeds from the pipeline/preprocessing_pipeline, which contains modular functions for each preprocessing step, such as handling missing values and encoding categorical variables.

    Parameters:
    data (pd.DataFrame): The input DataFrame to preprocess.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """

    # 0 - copy data
    log_preprocessing_step("Copying input data", logger=logger)
    data = data_original.copy()

    # 1 - Drop columns
    log_preprocessing_step("Dropping irrelevant columns", logger=logger)
    if cols_to_drop is not None:
        data = data.drop(columns=cols_to_drop, axis = 1)

    # 2 - Handle data inconsistencies and invalid values
    log_preprocessing_step("Forcing incoherent values to null", logger=logger)
    data = force_incoherence_to_null(data)

    # 3 - Handle missing values
    log_preprocessing_step("Building imputers for missing values", logger=logger)


    # to get the dict below run:
    # def get_missing_columns(data: pd.DataFrame) -> dict[str, str]:
    # """
    # Return the name and dtype of all columns with missing values.

    # Parameters
    # ----------
    # data : pd.DataFrame
    #     Input dataframe to inspect.

    # Returns
    # -------
    # dict[str, str]
    #     Mapping of {column_name: dtype_string} for columns with any NaN.
    # """
    # has_missing = data.isnull().any()
    # missing_cols = has_missing[has_missing].index

    # return {col: str(data[col].dtype) for col in missing_cols}
    #     method_map = {
    #     col: "knn" if dtype == "float64" else "most_frequent"
    #     for col, dtype in missing.items()
    # }

    data = impute_columns(data, dict = {
    'CARD_PROM_12': 'knn',
    'CHILDREN': 'knn',
    'DONOR_AGE': 'knn',
    'DONOR_GENDER': 'most_frequent',
    'FILE_CARD_GIFT': 'knn',
    'FREQUENCY_STATUS_97NK': 'knn',
    'HOME_OWNER': 'most_frequent',
    'INCOME_GROUP': 'knn',
    'LAST_GIFT_AMT': 'knn',
    'LIFETIME_CARD_PROM': 'knn',
    'LIFETIME_GIFT_AMOUNT': 'knn',
    'LIFETIME_GIFT_COUNT': 'knn',
    'LIFETIME_MAX_GIFT_AMT': 'knn',
    'LIFETIME_MIN_GIFT_AMT': 'knn',
    'LIFETIME_PROM': 'knn',
    'MEDIAN_HOME_VALUE': 'knn',
    'MEDIAN_HOUSEHOLD_INCOME': 'knn',
    'MONTHS_SINCE_FIRST_GIFT': 'knn',
    'MONTHS_SINCE_LAST_GIFT': 'knn',
    'MONTHS_SINCE_LAST_PROM_RESP': 'knn',
    'NUMBER_PROM_12': 'knn',
    'PCT_ATTRIBUTE1': 'knn',
    'PCT_ATTRIBUTE2': 'knn',
    'PCT_ATTRIBUTE3': 'knn',
    'PCT_ATTRIBUTE4': 'knn',
    'PCT_OWNER_OCCUPIED': 'knn',
    'PEP_STAR': 'knn',
    'PER_CAPITA_INCOME': 'knn',
    'RECENCY_STATUS_96NK': 'most_frequent',
    'RECENT_AVG_CARD_GIFT_AMT': 'knn',
    'RECENT_AVG_GIFT_AMT': 'knn',
    'RECENT_CARD_RESPONSE_COUNT': 'knn',
    'RECENT_CARD_RESPONSE_PROP': 'knn',
    'RECENT_RESPONSE_COUNT': 'knn',
    'RECENT_RESPONSE_PROP': 'knn',
    'RECENT_STAR_STATUS': 'knn',
    'SES': 'most_frequent',
    'URBANICITY': 'most_frequent'
    })

    # 4 - Handle Outliers

    log_preprocessing_step("Detecting outlier columns", logger=logger)
    outlier_columns_considered = (
        detect_outlier_columns(data, IQR_value)
        if outlier_columns is None
        else outlier_columns
    )
   
  
    log_preprocessing_step("Handling outliers", logger=logger)
    if outlier_method == "rescale":
        for col in outlier_columns_considered:
            data[col] = rescale_outliers(data=data[col], method="transform")
    elif outlier_method == "split":
        split_outlier_cluster(data=data, split_by=outlier_columns_considered, evasive=False)
        
    # 5 - Encode variables
    log_preprocessing_step("Encoding categorical variables", logger=logger)
    if encode == "onehot":
        data = one_hot_encode(data)
    elif encode == "label":
        data = label_encode(data)

    log_preprocessing_step("Preprocessing complete", logger=logger)
    return data
