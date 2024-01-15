import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple
from .dt import *

"""
Defines and exposes the API functions to client programs. 
Recall that the demo pipeline is split into three functions:
1. pipeline_train
2. pipeline_sliceline
3. pipeline_dt
There are also _py and _dml variants of each function. 
"""

def pipeline_train_py(
    train_x: pd.DataFrame,
    train_y: npt.NDArray[np.float64],
    test_x: pd.DataFrame,
    test_y: npt.NDArray[np.float64],
    max_depth: int
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    """
    params:
        train_x, train_y: Training data in "train" format.
        test_x, test_y: Test set in "train" format.
        max_depth: The maximum depth of slices to be considered for the slice_losses
                   return value.
    returns:
        train_x: Train set in "train" format. 
        train_losses: 1D array, train set losses. 
        train_stats: A dictionary to store various statistics about training
            session. It should at least contain the following key-value pairs:
            time_train (float): Time, in seconds, that training took. 
            time_func (float): Time, in seconds, that this entire function took. 
            agg_loss (float): Average loss (or accuracy) across all data points. 
            slice_losses (pd.DataFrame): A dataframe containing all slices up to
                max_depth in "DT" format, and additional "loss" column which
                stores average loss (or accuracy) for that slice, and "size"
                column which stores the size of that slice in train set. 
    """
    return train_x_binned, train_losses, train_stats

def pipeline_sliceline_py(
    train_x: pd.DataFrame,
    train_losses: npt.NDArray[np.float64],
    alpha: float,
    max_l: int,
    min_sup: int,
    k: int
) -> Tuple[pd.DataFrame, dict]:
    """
    params:
        train_x: Train set in "train" format. 
        train_losses: Train losses as 1D array. 
        alpha, max_l, min_sup, k: Standard sliceline parameters. 
    returns:
        slices: Top slices in "DT" format. It should also contain the additional
            columns "loss", "size", and "score". 
        sliceline_stats: A dictionary to store various statistics about the
            run of sliceline. It should contain the following key-value pairs:
            time_sliceline (float): Time, in seconds, that sliceline took. 
            time_func (float): Time, in seconds, that this entire function took. 
    """
    return slices, sliceline_stats

def pipeline_dt_py(
    sources: List[dt.DBSource],
    groups: pd.DataFrame,
    costs: npt.NDArray,
    query_counts: npt.NDArray,
    undersample_method: str,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    algos: List[str],
) -> dict:
    """
    params:
        sources: A list of DBSource objects that contain data sources metadata. 
        groups: A dataframe in "DT" format of the top slices. 
        costs: Cost model for each source. 
        query_counts: Query count for each slice. 
        undersample_method: Either "random", "medoids", or "none". Used to 
            undersample majority groups' excess data points. 
        train_x, train_y: Current train set in "train" format. 
        algos: A list of algorithms to run one after another. Valid values are:
            "random", "ratiocoll", "exploreexploit". 
    returns:
        A dictionary that maps from algorithm name to its results. 
        A valid key is a valid algorithm name, as explained above. 
        A valid value is a tuple (aug_x, aug_y, dt_stats): (pd.DataFrame, 
        pd.DataFrame, dict). aug_x and aug_y are train sets augmented with the
        additional data found. It MUST be shuffled. 
        dt_stats is a dictionary to store various statistics about the run of
        the DT algorithm specified in the key. It should contain the following
        key-value pairs:
            time (float): Time, in seconds, that DT took. 
            iters (int): The number of iterations that was required. 
            cost (float): The total cost required to satisfy query. 
            selected_sources (npt.NDArray): A sequence of sources that the DT
                algorithm chose. 
            selected_slices (npt.NDArray): A sequence of slice indices (based
                on the input)
    """
    return dt_results