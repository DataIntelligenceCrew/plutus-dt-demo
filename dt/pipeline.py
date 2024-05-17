import numpy as np
import numpy.typing as npt
import pandas as pd
import typing as tp
import time
from . import subroutines
from . import const
from . import utils
from . import dt

"""
Defines and exposes the API functions to client programs. 
Recall that the demo pipeline is split into three functions:
1. pipeline_train
2. pipeline_sliceline
3. pipeline_dt
There are also _py and _dml variants of each function. 
"""

def pipeline_train_py(
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_depth: int,
    task: str
) -> tp.Tuple[np.ndarray, dict]:
    """
    params:
        train: Training data in "raw" format.
        test: Test set in "raw" format.
        max_depth: The maximum depth of slices to be considered for the slice_losses return value. TODO: this does nothing atm
        task: Valid target task key. 
    returns:
        train_losses: 1D array, train set losses. 
        train_stats: A dictionary to store various statistics about training session. It should at least contain:
            time_train (float): Time, in seconds, that training took. 
            time_func (float): Time, in seconds, that this entire function took. 
            agg_loss (float): Average loss (or accuracy) across all data points. 
            slice_losses (pd.DataFrame): A dataframe containing all slices up to
                max_depth in "DT" format, and additional "loss" column which
                stores average loss (or accuracy) for that slice, and "size"
                column which stores the size of that slice in train set. 
    """
    time_start = time.time()
    # Convert dataset(s) to train format
    train = utils.recode_raw_to_train(train, task)
    test = utils.recode_raw_to_train(test, task)
    # Split dataset(s) to x, y
    y = const.Y_COLUMN[task]
    train_x, train_y = utils.split_df_xy(train, y)
    test_x, test_y = utils.split_df_xy(test, y)
    # Train model
    model = subroutines.pipeline_train_model(train_x, train_y, task)
    time_end_train = time.time()
    # Compute losses
    train_losses = subroutines.get_loss_vector(model, train_x, train_y, task)
    test_losses = subroutines.get_loss_vector(model, test_x, test_y, task)
    time_end_func = time.time()
    # Computer slice losses (train)
    bucket_train_x = utils.recode_raw_to_binned(train_x, task)
    slice_train_losses = []
    for col in bucket_train_x.columns:
        unique_vals = bucket_train_x[col].unique()
        for val in unique_vals:
            train_indices = bucket_train_x[bucket_train_x[col] == val].index
            print("DEBUG", val, train_indices, train_losses)
            train_subset_losses = train_losses[train_indices]
            if len(train_subset_losses) > 0:
                slice_train_losses.append(train_subset_losses.mean())
    # Computer slice losses (test)
    bucket_test_x = utils.recode_raw_to_binned(test_x, task)
    slice_test_losses = []
    for col in bucket_test_x.columns:
        unique_vals = bucket_test_x[col].unique()
        for val in unique_vals:
            test_indices = bucket_test_x[bucket_test_x[col] == val].index
            test_subset_losses = test_losses[test_indices]
            if len(test_subset_losses) > 0:
                slice_test_losses.append(test_subset_losses.mean())
    # Construct stats dictionary
    train_stats = {
        "time_train": time_end_train - time_start,
        "time_func": time_end_func - time_start,
        "agg_train_loss": np.mean(train_losses),
        "agg_test_loss": np.mean(test_losses),
        "slice_train_losses": sorted(slice_train_losses, reverse=True),
        "slice_test_losses": sorted(slice_test_losses, reverse=True)
    }
    return train_losses, test_losses, train_stats

def pipeline_sliceline_py(
    train: pd.DataFrame,
    train_losses: npt.NDArray[np.float64],
    alpha: float,
    max_l: int,
    min_sup: int,
    k: int,
    task: str
) -> tp.Tuple[npt.NDArray, dict]:
    """
    params:
        train: Train set in "raw" format. 
        train_losses: Train losses as 1D array. 
        alpha, max_l, min_sup, k: Standard sliceline parameters. 
    returns:
        slices: Top slices as numpy matrix, allowing for None values, ordered 
            by score from highest to kth highest. 
        sliceline_stats: A dictionary to store various statistics about the
            run of sliceline. It should contain the following key-value pairs:
            time_sliceline (float): Time, in seconds, that sliceline took. 
            time_func (float): Time, in seconds, that this entire function took. 
            scores: list of scores of slices (in same order)
            sizes: list of sizes of slices (in same order)
            errors: list of average errors of slices (in same order)
    """
    time_start = time.time()
    binned = utils.recode_raw_to_binned(train, task)
    binned_x = binned.drop(const.Y_COLUMN[task], axis=1)
    time_start_sliceline = time.time()
    slices, slices_stats = subroutines.get_slices(binned_x, train_losses, alpha, k, max_l, min_sup)
    time_end_sliceline = time.time()
    sliceline_stats = {
        "time_sliceline": time_end_sliceline - time_start_sliceline,
        "time_func": time_end_sliceline - time_start,
        "scores": [slice_['slice_score'] for slice_ in slices_stats],
        "sizes": [slice_['slice_size'] for slice_ in slices_stats],
        "errors": [slice_['slice_average_error'] for slice_ in slices_stats]
    }
    return slices, sliceline_stats

def pipeline_dt_py(
    slices: npt.NDArray,
    costs: npt.NDArray,
    query_counts: npt.NDArray,
    undersample_methods: tp.Union[str, tp.List[str]],
    train: pd.DataFrame,
    algos: tp.List[str],
    explore_scale: float,
    gt_stats: npt.NDArray,
    task: str
) -> dict:
    """
    params:
        sources: A list of DBSource objects that contain data sources metadata. 
        slices: Top k slices encoded as ndarray allowing None values. 
        costs: Cost model for each source. 
        query_counts: Query count for each slice. 
        undersample_methods: Either "random", "medoids", or "none". Used to 
            undersample majority groups' excess data points. I{
            time: 
        }f List, then
            the undersampling methods are used one at a time. 
        train: Current train set (both X and y) in "train" format. 
        algos: A list of algorithms to run one after another. Valid values are:
            "random", "ratiocoll", "exploreexploit". 
        task: Valid task key. 
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
    result = dict()
    train_x, train_y = utils.split_df_xy(train, const.Y_COLUMN[task])
    dt_ = dt.DT(slices, costs, train_x, explore_scale, gt_stats, task)
    if type(undersample_methods) is str:
        undersample_methods = [undersample_methods for _ in range(len(algos))]
    for idx, algo in enumerate(algos):
        additional_datasets, dt_stats = dt_.run(algo, query_counts)
        undersample_method = undersample_methods[idx]
        for idx, add in enumerate(additional_datasets):
            additional_datasets[idx] = utils.undersample(add, undersample_method, query_counts[idx])
        combined_data = pd.concat(additional_datasets, ignore_index=True)
        #aug_x, aug_y = utils.split_df_xy(combined_data, const.Y_COLUMN[task])
        result.update({algo: (combined_data, dt_stats)})
    return result

def pipeline_train_dml(
    train_x: pd.DataFrame,
    train_y: npt.NDArray[np.float64],
    test_x: pd.DataFrame,
    test_y: npt.NDArray[np.float64],
    max_depth: int
) -> tp.Tuple[np.ndarray, dict]:
    """
    params:
        train_x, train_y: Training data in "raw" format.
        test_x, test_y: Test set in "raw" format.
        max_depth: The maximum depth of slices to be considered for the slice_losses
                   return value.
    returns:
        train_losses: 1D array, train set losses. 
        train_stats: A dictionary to store various statistics about training
            session. It should at least contain the following key-value pairs:
            time_train (float): Time, in seconds, that training took. 
            time_func (float): Time, in seconds, that this entire function took. 
            agg_loss (float): Average loss (or accuracy) across all data points. 
            slice_losget_source_size(self.task, source)
                stat_tracker[source] /ses (pd.DataFrame): A dataframe containing all slices up to
                max_depth in "DT" format, and additional "loss" column which
                stores average loss (or accuracy) for that slice, and "size"
                column which stores the size of that slice in train set. 
    """
    # TODO: DML Implementation & call Python binding
    return train_losses, train_stats

def pipeline_sliceline_dml(
        train_sl: pd.DataFrame,
        train_losses: npt.NDArray[np.float64],
        alpha: float,
        max_l: int,
        min_sup: int,
        k: int,
        task: str
) -> tp.Tuple[pd.DataFrame, dict]:
    """
    params:
        train_x: Train set in "raw" format. 
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
    
    time_start = time.time()
    binned = utils.recode_raw_to_binned(train_sl.copy(), task) + 1
    binned_x = binned.drop(const.Y_COLUMN[task], axis=1)

    if task == 'flights-classify':
        train_losses = [1 if x == 0 else 0 for x in train_losses]
        train_losses = pd.Series(train_losses).to_numpy()

    time_start_sliceline = time.time()
    slices, slices_stats = subroutines.get_slices_dml(binned_x, train_losses, alpha, k, max_l, min_sup)
    
    time_end_sliceline = time.time()
    sliceline_stats = {
        "time_sliceline": time_end_sliceline - time_start_sliceline,
        "time_func": time_end_sliceline - time_start,
        "scores": [slice_[0] for slice_ in slices_stats],
        "sizes": [slice_[3] for slice_ in slices_stats],
        "errors": [slice_[1] for slice_ in slices_stats]
    }
    
    return slices, sliceline_stats

def pipeline_dt_dml(
    groups: pd.DataFrame,
    costs: npt.NDArray,
    query_counts: npt.NDArray,
    undersample_method: str,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    algos: tp.List[str],
) -> dict:
    """
    params:
        sources: A list of DBSource objects that contain data sources metadata. 
        groups: A dataframe in "DT" format of the top slices. 
        costs: Cost model for each source. 
        query_counts: Query count for each slice. 
        undersample_method: Either "random", "medoids", or "none". Used to 
            undersample majority groups' excess data points. 
        train_x, train_y: Current train set in "raw" format. 
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
    # TODO: DML Implementation & call Python binding
    return dt_results
