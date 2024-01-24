import re
import csv
import itertools
import math
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import typing as tp
import random
from . import utils
from . import dbsource
from . import const
from . import subroutines

"""
Represents a single instance of the DT problem. 
In other words, implements (D, G, C, Q) as a class. 
Implements the random, Ratiocoll, and EpsilonGreedy algorithms. 
Supports integer-valued CSV files only for the time being. 
"""
class DT:
    def __init__(self,
        slices: npt.NDArray,
        costs: npt.NDArray,
        train_x: pd.DataFrame,
        explore_scale: float,
        task: str):
        """
        params
            sources: A list of CSV filenames
            costs: A list of floating point cost for each source
            slices: A list of slices encoded as numpy matrix
            task: Valid task key string. 
        """
        self.n = const.SOURCES[task]['n'] # Number of sources
        self.costs = np.array(costs) # costs
        self.slices = slices # Slices
        self.m = len(slices) # Number of slices
        self.task = task
        self.explore_scale = explore_scale
        self.train_x = train_x

    def __repr__(self):
        s = "n: " + str(self.n) + ", m: " + str(self.m) + "\n"
        s += "costs: " + str(self.costs) + "\n"
        s += "slices: " + str(self.slices) + "\n"
        return s
        
    def run(self, algorithm: str, query_counts: npt.NDArray) -> tp.Tuple[tp.List[pd.DataFrame], dict]:
        """
        params:
            algorithm: either 'ratiocoll', 'exploreexploit', or 'random' for now
            query_counts: m-length vector denoting minimum query counts
        returns: 
            1. Additional data acquired for each slice, in same order. Encoded
                in raw format.
            2. DT stats dictionary. 
        """
        additional_data = [None] * self.m
        remaining_query = np.copy(query_counts)
        total_cost = 0.0
        # Initialize stat tracker
        if algorithm == "ratiocoll":
            binned_x = utils.recode_raw_to_binned(self.train_x, self.task).to_numpy()
            stat_tracker = np.empty((self.n, self.m))
            for source in range(self.n):
                for slice_ in range(self.m):
                    cnt = dbsource.get_slice_count(self.task, source, self.slices[slice_])
                    stat_tracker[source][slice_] = cnt
                source_cnt = dbsource.get_source_size(self.task, source)
                stat_tracker[source] /= source_cnt
        elif algorithm == "exploreexploit":
            stat_tracker = np.zeros((self.n, self.m), dtype=int)
            # Compute the number of exploration iterations
            Q = np.sum(query_counts) * self.explore_scale
            explore_iters = math.ceil((0.5 * math.pow(Q, 2/3)) / const.SOURCES[self.task]['batch'])
            # Round up explore_iters to ensure uniform exploration
            self.explore_iters = math.ceil(explore_iters / self.n) * self.n
        elif algorithm == "random":
            stat_tracker = None
        print("Stats:", stat_tracker)
        
        itr = 0
        # Loop while query is not satisfied
        while np.any(remaining_query > 0):
            chosen_source = self.choose_ds(algorithm, itr, stat_tracker, remaining_query)
            itr += 1
            # Issue query and recode result to raw
            query_result = dbsource.get_query_result(self.task, chosen_source)
            query_result = utils.recode_db_to_raw(query_result, self.task)
            # Keep track of stats
            total_cost += self.costs[chosen_source] * len(query_result)
            # Split query result to x, y
            result_x, _ = utils.split_df_xy(query_result, const.Y_COLUMN[self.task])
            binned_x = utils.recode_raw_to_binned(result_x, self.task).to_numpy()
            # slice_ownership[i][j] denotes whether ith tuple belongs to slice j
            slice_ownership = subroutines.slice_ownership(binned_x, self.slices)
            # Count the frequency of each subgroup in query result
            for i in range(len(result_x)):
                for j in range(self.m):
                    # Row belongs to slice
                    if slice_ownership[i][j]:
                        # Update stats if we're running such an algorithm
                        if algorithm == "exploreexploit":
                            stat_tracker[chosen_source][j] += 1
                        # Decrement remaining query
                        remaining_query[j] -= 1
                        # Append row to additional data
                        result_row = query_result.iloc[i,:].to_frame().T
                        if additional_data[j] is None:
                            additional_data[j] = result_row
                        else:
                            additional_data[j] = pd.concat([additional_data[j], result_row], ignore_index=True)
        
        # Generate stats
        stats = {
            "cost": total_cost
        }
        print("DT ", algorithm, " complete")
        return additional_data, stats
    
    def choose_ds(self, algorithm: str, itr: int, stat_tracker, remaining_query: npt.NDArray) -> int:
        if algorithm == "random":
            ds = random.choice(range(self.n))
            print(algorithm, ds)
            return ds
        elif algorithm == "exploreexploit":
            if itr < self.explore_iters:
                priority_source = itr % self.n
                print(algorithm, priority_source)
                return priority_source
            else:
                P = np.maximum(stat_tracker / max(1, np.sum(stat_tracker)), np.full(np.shape(stat_tracker), const.EPSILON_PROB))
                C_over_P = np.reshape(self.costs.T, (self.n, 1)) / P
                min_C_over_P = np.amin(C_over_P, axis=0)
                group_scores = remaining_query * min_C_over_P
                priority_group = np.argmax(group_scores)
                priority_source = np.argmin(C_over_P[:,priority_group], axis=0)
                print(algorithm, priority_source)
                return priority_source
        elif algorithm == "ratiocoll":
            P = np.maximum(stat_tracker / max(1, np.sum(stat_tracker)), np.full(np.shape(stat_tracker), const.EPSILON_PROB))
            C_over_P = np.reshape(self.costs.T, (self.n, 1)) / P
            min_C_over_P = np.amin(C_over_P, axis=0)
            group_scores = remaining_query * min_C_over_P
            priority_group = np.argmax(group_scores)
            priority_source = np.argmin(C_over_P[:,priority_group], axis=0)
            print(algorithm, priority_source)
            return priority_source