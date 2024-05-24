import numpy as np
import numpy.typing as npt
import pandas as pd
import typing as tp
import xgboost as xgb


# Splits a dataframe into xs and ys based on the given name of y variable
def split_df_xy(df: pd.DataFrame, y: str) -> tp.Tuple[pd.DataFrame, npt.NDArray]:
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


def get_loss_vector(
        model: tp.Union[xgb.XGBRegressor, xgb.XGBClassifier],
        data_x: pd.DataFrame,
        data_y: npt.NDArray,
        loss_name: str
) -> npt.NDArray:
    """
    params:
        model: Trained XGBoost model.
        data_x: Train or test data in train encoding.
        data_y: Train or test data in train encoding.
        loss_name: Method for computing loss. 'binary' assigns a loss of 1 if incorrect and 0 if correct. 'square'
                   squares the error between gt and pred
    returns: A 1D loss vector, where loss is square loss if task is regression, and 1 or 0 if task is classification.
    """
    preds = np.ndarray(model.predict(data_x))
    if loss_name == 'binary':
        return np.not_equal(preds, data_y).astype(float)
    elif loss_name == 'square':
        return (data_y - preds) ** 2
    else:
        raise ValueError('Unsupported loss type.')


def get_top_level_slice_losses(binned_x: pd.DataFrame, losses: npt.NDArray) -> tp.List[float]:
    slice_losses = []
    for col in binned_x.columns:
        unique_vals = binned_x[col].unique()
        for val in unique_vals:
            train_indices = binned_x[binned_x[col] == val].index
            train_subset_losses = losses[train_indices]
            if len(train_subset_losses) > 0:
                slice_losses.append(train_subset_losses.mean())
    return slice_losses

