import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple
import dbsource
import const

# Splits a dataframe into xs and ys based on the given name of y variable
def split_df_xy(df: pd.DataFrame, y: str) -> Tuple[pd.DataFrame, npt.NDArray]:
    """
    params:
        df: A dataset in any encoding. 
        y: The name of the y variable in the dataframe. 
    returns:
        df_x: Dataframe containing all but the y variable. 
        df_y: NDArray containing the y variable's values. 
    """
    df_y = df[y]
    df_x = df.drop(y, axis=1)
    return df_x, df_y

"""
Dataset encoding conversion methods. 
1. db: The encoding returned directly from the database query. 
2. raw: The encoding w/ only basic processing. Same as db, except: 
    - Only features defined in const.py are selected, others discarded
    - All categorical features are integer-encoded
3. train: Encoding used for training XGBoost. Same as raw, except:
    - Columns defined in DUMMY_FEATURES need to be one-hot encoded. 
4. binned: Encoding used for sliceline. Same as raw, except:
    - Numeric features need to be integer encoded. 

The following recoding directions are supported:
1. db -> raw
2. raw -> train
3. raw -> binned
"""

def recode_db_to_raw(db_data: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    params: 
        db_data: Dataset in DB encoding. 
        task: Valid target task key. 
    returns: Dataset in raw encoding. 
    """
    # Grab the constants
    y_column = const.Y_COLUMN[task]
    cat_features = const.CATEGORICAL_FEATURES[task]
    num_features = const.NUMERIC_FEATURES[task]
    cat_mappings = const.CATEGORICAL_MAPPINGS[task]
    # Select only the specified features
    all_features = cat_features + num_features + [y_column]
    df = db_data[all_features]
    # Integer encode all categorical features
    df[cat_features] = df[cat_features].replace(cat_mappings)
    # If y is categorical, also integer encode y
    if const.Y_IS_CATEGORICAL[task]:
        df[y_column] = df[y_column].replace(cat_mappings[y_column])
    return df

def recode_raw_to_train(raw_data: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    params:
        raw_data: Dataset in raw encoding. 
        task: Valid target task key. 
    returns: Dataset in train encoding. 
    """
    # Dummy encode specified features
    return pd.get_dummies(raw_data, columns=const.DUMMY_FEATURES[task])

def recode_raw_to_binned(raw_data: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    params:
        raw_data: Dataset in raw encoding. 
        task: Valid target task key. 
    returns: Dataset in binned encoding. 
    """
    # Bin numeric features
    num_features = const.NUMERIC_FEATURES[task]
    num_bins = const.NUMERIC_BIN_BORDERS[task]
    for num in num_features:
        raw_data[num] = pd.cut(raw_data[num], bins = num_bins[num], labels=None)
    return raw_data

def undersample(dataset: pd.DataFrame, method: str) -> pd.DataFrame:
    return None

# Testing methods
if __name__ == "__main__":
    task = "flights-classify"
    raw_data = dbsource.get_query_result(task, 2)
    print("Data in raw format:")
    print(raw_data)
    print()

    train_data = recode_raw_to_train(raw_data, task)
    print("Data in train format:")
    print(train_data)
    print()

    binned_data = recode_raw_to_binned(raw_data, task)
    print("Data in binned format:")
    print(binned_data)
    print()