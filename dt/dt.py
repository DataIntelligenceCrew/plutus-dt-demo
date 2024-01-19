import re
import csv
import itertools
import math
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .dbsource import DBSource
from typing import List

"""
Represents a single instance of the DT problem. 
In other words, implements (D, G, C, Q) as a class. 
Implements the random, Ratiocoll, and EpsilonGreedy algorithms. 
Supports integer-valued CSV files only for the time being. 
"""
class DT:
    def __init__(self,
        slices: npt.NDArray,
        costs: npt.NDArray
        task: str):
        """
        params
            sources: A list of CSV filenames
            costs: A list of floating point cost for each source
            slices: A list of slices encoded as numpy matrix
            task: Valid task key string. 
        """
        self.n = len(sources) # Number of sources
        self.costs = np.array(costs) # costs
        self.slices = slices # Slices
        self.m = len(slices) # Number of slices
        self.task = task

    def __repr__(self):
        s = "n: " + str(self.n) + ", m: " + str(self.m) + "\n"
        s += "costs: " + str(self.costs) + "\n"
        s += "slices: " + str(self.slices) + "\n"
        return s
        
    def run(self, algorithm: str, query_counts: npt.NDArray) -> Tuple[List[pd.DataFrame], dict]:
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
        # Initialize stat tracker
        match algorithm:
            case "ratiocoll":
                stat_tracker = 
            case "exploreexploit":
                stat_tracker = np.zeros((self.n, self.m), dtype=int)
            case "random":
                pass

        match algorithm:
            case "ratiocoll":
                return self.ratiocoll(query_counts)
            case "exploreexploit":
                return self.explore_exploit(query_counts)
            case "random":
                return self.random_baseline(query_counts)
    
    def explore_exploit(self, query_counts: npt.NDArray) -> Tuple[List[pd.DataFrame], dict]:
        Q = sum(query_counts) # Total query requirement
        additional_data = [None] * self.m # Separate collected dataset per slice
        remaining_query = np.copy(query_counts)
        stat_tracker = np.zeros((self.n, self.m), dtype=int)

        # Compute the number of exploration iterations
        explore_iters = math.ceil((0.5 * math.pow(Q, 2/3)) / self.batch)
        # Round up explore_iters to ensure uniform exploration
        explore_iters = math.ceil(explore_iters / self.n) * self.n

        i = 0
        while np.any(remaining_query > 0):
            # Choose the priority source
            if i < explore_iters: # Explore sources uniformly
                priority_source = i % self.n
            else:
                # Ensure that probs aren't zero to prevent numerical issues
                P = np.maximum(stat_tracker / np.sum(stat_tracker, axis=1), 
                    const.EPSILON_PROB)
                C_over_P = np.reshape(self.costs.T, (self.n, 1)) / P
                min_C_over_P = np.amin(C_over_P, axis=0)
                group_scores = remaining_query * min_C_over_P
                priority_group = np.argmax(group_scores)
                priority_source = np.argmin(C_over_P[:,priority_group], axis=0)
            # Get query result
            query_result = self.sources[priority_source].get_query_result()


            # Split query result to x, y
            result_x, result_y = split_xy(query_result, self.sources[priority_source].y_name)
            result_x = result_x
            result_y = list(result_y)
            # Count the frequency of each subgroup in query result
            for i in range(len(result_x)):
                result_x_row = result_x.iloc[i,:].to_frame().T
                #if i < 4:
                #	print("xi", result_x_row, type(result_x_row))
                result_y_row = result_y[i]
                slices = self.slice_ownership(result_x_row)
                #print(result_row, slices)
                self.stats[priority_source][slices] += 1
                if len(slices) > 0:
                    unified_set = result_x_row if unified_set is None else pd.concat([unified_set, result_x_row], ignore_index=True)
                    unified_ys.append(result_y_row)
                remaining_query[slices] -= 1
                remaining_query = np.maximum(remaining_query, 0)
                if not np.any(remaining_query > 0):
                    break
            i += 1
        unified_set['median_house_value'] = unified_ys
        return unified_set
        
    def ratiocoll(self, query_counts):
        """
        params:
            query_coutns: (1, m) ndarray denoting query counts for each group
            batch: int, batch size for reading sources
            discard: whether to keep or discard excess tuples
        """
        # The actual collected set
        unified_xs_df = pd.DataFrame()
        unified_ys = []
        remaining_query = np.copy(query_counts)

        # Precompute some matrices
        # (n, m) matrix, where P has been normalized by C
        P = self.stats / 100000
        C_over_P = np.reshape(self.costs.T, (self.n,1)) / P
        # (1, m) matrix, where we find the minimum C/P for each group
        min_C_over_P = np.amin(C_over_P, axis=0)

        query_times = 0
        while np.any(remaining_query > 0):
            query_times += 1
            # Score for each group, (1, m)
            group_scores = remaining_query * min_C_over_P
            #print("group scores:", group_scores)
            # Priority group & source
            priority_group = np.argmax(group_scores)
            #print("priority group:", priority_group)
            priority_source = np.argmin(C_over_P[:,priority_group], axis=0)
            #print("priority source:", priority_source)
            # Batch query chosen source
            if type(self.sources[0]) == str: # Sources are CSV files
                query_result = np.array(self.readers[priority_source].next())
            else: # Sources are DB queries
                query_result = np.array(self.sources[priority_source].get_query_result())
            # Count the frequency of each subgroup in query result
            for b, result_row in enumerate(query_result):
                slices = self.slice_ownership(result_row)
                print(b, result_row, slices)
                if len(slices) > 0:
                    unified_xs_df.append(result_row)
                remaining_query[slices] -= 1
                remaining_query = np.maximum(remaining_query, 0)
                if not np.any(remaining_query > 0):
                    break
        unified_set = np.array(unified_set)
        #print(unified_set, len(unified_set), query_times)
        return unified_set
    
    def slice_ownership(self, result_row):
        """
        returns a boolean list denoting the slices the other result_row belongs to
        assumes result_row is a single-row DF
        """
        ownership = []
        for slice_ in self.slices:
            ownership.append(belongs_to_slice(slice_, result_row.values.tolist()[0]))
        return ownership
