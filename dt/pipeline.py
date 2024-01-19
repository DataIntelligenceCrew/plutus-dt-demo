import numpy as np
import numpy.typing as npt
import pandas as pd
import typing as tp
import dbsource
import subroutines
import time
import const
import utils

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
        max_depth: The maximum depth of slices to be considered for the slice_losses
                   return value.
        task: Valid target task key. 
    returns:
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
    # Construct stats dictionary
    train_stats = {
        "time_train": time_end_train - time_start,
        "time_func": time_end_func - time_start,
        "agg_train_loss": np.mean(train_losses),
        "agg_test_loss": np.mean(test_losses),
        "slice_losses": None # TODO: Figure out how to implement this
    }
    return train_losses, train_stats

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
    dt = dt.DT(slices, costs, task)
    if type(undersample_methods) is str:
        undersample_methods = [undersample_method] * len(algos)
    for idx, algo in enumerate(algos):
        additional_datasets, dt_stats = dt.run(algo, query_counts)
        undersample_method = undersample_methods[idx]
        for idx, add in enumerate(additional_datasets):
            additional_datasets[idx] = utils.undersample(additional_data, undersample_method)
        combined_data = None # TODO: Combine all undersampled dataframed
        aug_x, aug_y = split_df_xy(combined_data, const.Y_COLUMN[y_name])
        result.update({algo: (aug_x, aug_y, dt_stats)})
    return dt_results

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
            slice_losses (pd.DataFrame): A dataframe containing all slices up to
                max_depth in "DT" format, and additional "loss" column which
                stores average loss (or accuracy) for that slice, and "size"
                column which stores the size of that slice in train set. 
    """
    # TODO: DML Implementation & call Python binding
    return train_losses, train_stats

def pipeline_sliceline_dml(
    train_x: pd.DataFrame,
    train_losses: npt.NDArray[np.float64],
    alpha: float,
    max_l: int,
    min_sup: int,
    k: int
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
    # TODO: DML Implementation & call Python binding
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

# Test cases
if __name__ == "__main__":
    task = "flights-regress"
    y_str = const.Y_COLUMN[task]

    train = dbsource.get_train(task)
    print("Train set:")
    print(train)
    print()

    test = dbsource.get_test(task)
    print("Test set:")
    print(test)
    print()

    train_losses, train_stats = pipeline_train_py(train, test, 1, task) 

    print("Train losses:")
    print(train_losses)
    print()

    print("Train stats:")
    print(train_stats)
    print()

    slices, sliceline_stats = pipeline_sliceline_py(train, train_losses, 0.8, 2, 0, 3, task)

    print("Slices:")
    print(slices)
    print()

    print("Sliceline stats:")
    print(sliceline_stats)
    print()