import numpy as np
import numpy.typing as npt
import pandas as pd
import typing as tp
import xgboost as xgb
from sliceline.slicefinder import Slicefinder
from . import const
from . import dbsource

"""
Subroutines called by the _py pipeline API functions. 
"""

def pipeline_train_model(
    train_x: pd.DataFrame,
    train_y: npt.NDArray,
    task: str
) -> tp.Union[xgb.XGBRegressor, xgb.XGBClassifier]:
    """
    params:
        train_x: Training data in train encoding.
        train_y: Training data in train encoding.
        task: Valid target task key. 
    """
    if const.Y_IS_CATEGORICAL[task]:
        model = xgb.XGBClassifier()
    else:
        model = xgb.XGBRegressor()
    print("pipeline - train model")
    print(train_x, train_y)
    print(train_x.dtypes)
    print(set(train_x['origin_state']))
    # Note that for Python xgboost library, enable_categorical = True is reuquired
    # to deal with categorical data w/o one-hot encoding
    model.fit(train_x, train_y)
    return model

def get_loss_vector(
    model: tp.Union[xgb.XGBRegressor, xgb.XGBClassifier],
    data_x: pd.DataFrame,
    data_y: npt.NDArray,
    task: str
) -> npt.NDArray:
    """
    params:
        model: Trained XGBoost model. 
        data_x: Train or test data in train encoding. 
        data_y: Train or test data in train encoding. 
        task: Valid target task key. 
    returns: A 1D loss vector, where loss is square loss if task is regression,
        and 1 or 0 if task is classification. 
    """
    preds = model.predict(data_x)
    if const.Y_IS_CATEGORICAL[task]:
        loss = np.abs(data_y - preds)
        return loss[loss > 0] == 1
    else:
        return (data_y - preds)**2

def get_slices(binned_x, losses, alpha, k, max_l, min_sup) -> tp.Tuple[npt.NDArray, dict]:
    """
    params:
        binned_x: X variables of training data in binned encoding. 
        losses: 1D loss vector from the training data. 
        alpha: Parameter for prioritizing size or performance, from 0.0 to 1.0
        k: The number of slices to return. 
        max_l: The maximum level of slices to search. 
        min_up: Minimum number of tuples required per slice to return. 
    returns: 
        slices: Top-k slices encoded as (d,k) integer matrix (allows None). 
        slices_statistics: Same as the sliceline package. 
            top_slices_statistics_: list of dict of length `len(top_slices_)`
            The statistics of the slices found sorted by slice's scores.
            For each slice, the following statistics are stored:
                - slice_score: the score of the slice (defined in `_score` method)
                - sum_slice_error: the sum of all the errors in the slice
                - max_slice_error: the maximum of all errors in the slice
                - slice_size: the number of elements in the slice
                - slice_average_error: the average error in the slice (sum_slice_error / slice_size)
    """
    print("Running sliceliner with alpha =", alpha, ", k", k, ", max_l", max_l, ", min_sup", min_sup)
    sf = Slicefinder(alpha = alpha, k = k, max_l = max_l, min_sup = min_sup, verbose = True)
    print("Fitting sliceliner with", binned_x, losses)
    print(binned_x)
    print(binned_x.dtypes)
    print(set(binned_x['day']))
    #print("Loss values: ", set(losses))
    sf.fit(binned_x.to_numpy(), losses)
    print("slicefinder obj", sf)
    print("top silces", sf.top_slices_)
    #print("feature names in", sf.feature_names_in_)
    print("feature names out", sf.get_feature_names_out()),
    print("top slices stats", sf.top_slices_statistics_)
    print("average error", sf.average_error_)
    #df = pd.DataFrame(sf.top_slices_, columns=sf.feature_names_in_, index=sf.get_feature_names_out())
    #print(df)
    return sf.top_slices_, sf.top_slices_statistics_

def belongs_to_slice(
    x: npt.NDArray,
    slice_: npt.NDArray
) -> bool:
    """
    params: 
        x: d-length numpy array encoding one data point. 
        slice: d-length numpy array encoding one slice. 
    """
    # Filter out None columns
    not_none_mask = slice_ != None
    not_none_x = x[not_none_mask]
    # Test equality of not-None columns
    not_none_slice = slice_[not_none_mask]
    return np.array_equal(not_none_x, not_none_slice)

def slice_ownership(
    binned_x: npt.NDArray, 
    slices: npt.NDArray
) -> npt.NDArray:
    """
    params:
        binned_x: X variables of data in binned encoding, as numpy matrix. 
        slices: Slices encoded as numpy matrix allowing None. 
    returns:
        A numpy boolean matrix where array[i][j] is set to True if the ith
        tuple belongs to the jth slice. 
    """
    n = len(binned_x)
    k = len(slices)
    ownership_mat = np.empty((n,k), dtype=bool)
    for i in range(n):
        for j in range(k):
            ownership_mat[i][j] = belongs_to_slice(binned_x[i], slices[j])
    print(ownership_mat)
    return ownership_mat

def get_slice_cnts(
    binned_x: npt.NDArray,
    slices: pd.DataFrame
) -> npt.NDArray:
    """
    params:
        binned_x: X variables of dataset in binned encoding. 
        slices: Top-k slices in integer dataframe format. 
    returns:
        The number of rows in binned_x that belong to the given slices. 
    """
    ownership_mat = slice_ownership(binned_x, slices)
    slice_cnts = np.sum(ownership_mat, axis=0)
    return slice_cnts