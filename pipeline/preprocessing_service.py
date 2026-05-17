import pandas as pd
from preprocessing_pipeline.preprocessing_cleaning_invalid import *
from preprocessing_pipeline.preprocessing_encoding import *
from preprocessing_pipeline.preprocessing_feature_engineering import *
from preprocessing_pipeline.preprocessing_missing_values import *
from preprocessing_pipeline.preprocessing_scaling import *
from preprocessing_pipeline.preproccessing_outliers import *



def preprocess_data(data_original: pd.DataFrame,
    outlier_columns = None,
    outlier_method = "rescale",
    IQR_value = 1.5,
    encode = None               ) -> pd.DataFrame:
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
    data = data_original.copy()
    # 1 - Handle data inconsistencies and invalid values

    data = force_incoherence_to_null(data)

    # 2 - Handle missing values
    numeric_imputer = build_numeric_imputer(strategy="median")
    categorical_imputer = build_categorical_imputer(strategy="most_frequent")

    ### use inputers
    outlier_columns_considered = [detect_outlier_columns(data, IQR_value ) if outlier_columns == None else outlier_columns ]
   
    # 3 - Handle Outliers
    if outlier_method == "rescale":
        for col in outlier_columns_considered:
            data[col] = rescale_outliers(data = data[col], method = "transform")
    elif outlier_method == "split" :
        data[col] = split_outlier_cluster(data = outlier_columns_considered, evasive= False)
        
    # 4 - Encode variables

    # 5 - Feature Engineering
        


    
