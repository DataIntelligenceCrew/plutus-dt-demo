import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple

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
    df_y = df[y_name]
    df_x = df.drop(y_name, axis=1)
    return df_x, df_y

"""
Dataset encoding conversion methods. There are three encodings used:
1. raw: The encoding returned directly from the database query. Assume that the
    db tables have gone through basic processing. This is the format used to
    pass around data in between pipeline functions. 
2. train: The encoding used for training. We will use XGBoost throughout the
    demo and so will follow the specifications for SystemDS' XGBoost script,
    which requires categorical variables to be one-hot encoded. 
3. binned: The encoding used for sliceline. Note that both Python and DML slice
    -line implementations are able to take in integer binned features, which is
    consistent with the paper. As such, all categorical and continuous features
    need to be integer encoded. One-hot encoding should NOT be done. 
Raw -> Train and Raw <-> Binned helper functions are implemented. 
"""

def recode_raw_to_train(raw_data: pd.DataFrame) -> pd.DataFrame:

def recode_raw_to_binned(raw_data: pd.DataFrame) -> pd.DataFrame:

def recode_binned_to_raw(binned_data: pd.DataFrame) -> pd.DataFrame:
