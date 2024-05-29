import time
from . import subroutines
from . import dt
from .task import *
from .source import *
import numpy.typing as npt

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
    print("Train X:")
    print(train_x)
    print("Train y:")
    print(train_y)
    test_x, test_y = split_df_xy(test, task.y_column_name())
    print("Test X:")
    print(test_x)
    print("Test y:")
    print(test_y)
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
    algos: tp.List[str],
    explore_scale: float,
) -> dict:
    """
    params;
        task: A valid instantiation of an AbstractTask subclass.
        slices: TOp k slices encoded as ndarray allowing None values.
        query_counts: Query count for each slice.
        algos: A list of algorithms to run one after another. Supports "random", "ratiocoll", "exploreexploit".
        explore_scale: Parameter alpha to adjust exploration rate for ExploreExploit.
        gt_stats: Ground truth statistics for the task.
    returns:
        A dictionary mapping from algorithm to its result.
        A valid value (result) is itself a dictionary with the following fields:
            data: Additional data, as a DataFrame.
            time: Time, in seconds, that the algorithm took.
            iters: Number of iterations that the algorithm took.
            cost: Total cost required to satisfy query.
            selected_sources: A sequence of sources that the algorithm chose.
            selected_slices: A sequence of slice indices (based on the input).
    """
    result = dict()
    dt_ = dt.DT(task, slices, explore_scale, 10)
    for algo in algos:
        print(algo)
        algo_result = dt_.run(algo, query_counts)
        result.update({algo: algo_result})
        print("FINISHED RUNNING ", algo)
    return result


def pipeline_sliceline_dml(
        task: AbstractTask,
        train_sl: pd.DataFrame,
        train_losses: npt.NDArray[np.float64],
        alpha: float,
        max_l: int,
        min_sup: int,
        k: int
) -> dict:
    """
    params:
        train_x: Train set in "raw" format. 
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
    binned = task.binning_for_sliceline(train_sl) + 1
    binned_x = binned.drop(task.y_column_name(), axis=1)

    slices, slices_stats = subroutines.get_slices_dml(binned_x, train_losses, alpha, k, max_l, min_sup)

    time_end = time.time()
    ret = {
        "slices": slices,
        "time": time_start - time_end,
        "scores": [slice_[0] for slice_ in slices_stats],
        "sizes": [slice_[3] for slice_ in slices_stats],
        "errors": [slice_[1] for slice_ in slices_stats]
    }

    return ret
