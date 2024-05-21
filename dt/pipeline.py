import numpy as np
import numpy.typing as npt
import pandas as pd
import typing as tp
import time
from . import subroutines
from . import const
from . import utils
from . import dt
from .task import *
from .source import *

"""
Defines and exposes the API functions to client programs. 
Recall that the demo pipeline is split into three functions:
1. pipeline_train
2. pipeline_sliceline
3. pipeline_dt
There are also _py and _dml variants of each function. 
"""


def pipeline_train_py(task: AbstractTask, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    params:
        task: A valid instantiation of an AbstractTask subclass.
    returns a dictionary with the following fields:
        train_losses (1D ndarray): Train set losses.
        test_losses (1D ndarray): Test set losses.
        time (float): Time, in seconds, that this function took.
        agg_train_loss (float): Average loss (or accuracy) across all train data points.
        agg_test_loss (float): Average loss (or accuracy) across all test data points.
        slice_train_losses, slice_test_losses (list of floats): Average loss of each top-level slice in train/test sets.
    """
    time_start = time.time()
    # Convert dataset(s) to train format
    train_x, train_y = split_df_xy(train, task.y_column_name())
    test_x, test_y = split_df_xy(test, task.y_column_name())
    # Train model
    model_name = 'xgboost-classify' if task.y_is_categorical else 'xgboost-regress'
    model = subroutines.pipeline_train_model(train_x, train_y, model_name)
    # Compute losses
    loss_name = 'binary' if task.y_is_categorical else 'square'
    train_losses = subroutines.get_loss_vector(model, train_x, train_y, loss_name)
    test_losses = subroutines.get_loss_vector(model, test_x, test_y, loss_name)
    # Compute slice losses
    slice_train_losses = subroutines.get_top_level_slice_losses(task.binning_for_sliceline(train_x), train_losses)
    slice_test_losses = subroutines.get_top_level_slice_losses(task.binning_for_sliceline(test_x), test_losses)
    # Construct the result dictionary
    time_end = time.time()
    result = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'time': time_end - time_start,
        'agg_train_loss': np.mean(train_losses),
        'agg_test_loss': np.mean(test_losses),
        'slice_train_losses': sorted(slice_train_losses, reverse=True),
        'slice_test_losses': sorted(slice_test_losses, reverse=True)
    }
    return result


def pipeline_sliceline_py(
    task: AbstractTask,
    train: pd.DataFrame,
    train_losses: npt.NDArray[np.float64],
    alpha: float,
    max_l: int,
    min_sup: int,
    k: int
) -> dict:
    """
    params:
        task: A valid instantiation of an AbstractTask subclass.
        train: Train set in "raw" format. 
        train_losses: Train losses as 1D array. 
        alpha, max_l, min_sup, k: Standard sliceline parameters. 
    returns a dictionary with the following fields:
        slices: Top slices as numpy matrix, allowing for None values, ordered descending.
        time (float): Time, in seconds, that this function took.
        scores: list of scores of slices (in same order)
        sizes: list of sizes of slices (in same order)
        errors: list of average errors of slices (in same order)
    """
    time_start = time.time()
    binned_x = task.binning_for_sliceline(train).drop(task.y_column_name(), axis=1)
    slices, slices_stats = subroutines.get_slices(binned_x, train_losses, alpha, k, max_l, min_sup)
    time_end = time.time()
    result = {
        "slices": slices,
        "time": time_end - time_start,
        "scores": [slice_['slice_score'] for slice_ in slices_stats],
        "sizes": [slice_['slice_size'] for slice_ in slices_stats],
        "errors": [slice_['slice_average_error'] for slice_ in slices_stats]
    }
    return result


def pipeline_dt_py(
    task: AbstractTask,
    slices: npt.NDArray,
    query_counts: npt.NDArray,
    undersample_methods: str,
    algos: tp.List[str],
    explore_scale: float,
) -> dict:
    """
    params;
        task: A valid instantiation of an AbstractTask subclass.
        slices: TOp k slices encoded as ndarray allowing None values.
        query_counts: Query count for each slice.
        undersample_methods: Currently only supports "random" and "none", used to undersample excess data.
        algos: A list of algorithms to run one after another. Supports "random", "ratiocoll", "exploreexploit".
        explore_scale: Parameter alpha to adjust exploration rate for ExploreExploit.
        gt_stats: Ground truth statistics for the task.
    returns:
        A dictionary mapping from algorithm to its result.
        A valid value (result) is itself a dictionary with the following fields:
            time: Time, in seconds, that the algorithm took.
            iters: Number of iterations that the algorithm took.
            cost: Total cost required to satisfy query.
            selected_sources: A sequence of sources that the algorithm chose.
            selected_slices: A sequence of slice indices (based on the input).
    """
    result = dict()
    dt_ = dt.DT(task, slices, explore_scale, 10)
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
