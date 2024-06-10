import random as rnd

import math

from . import utils
from .task import *

"""
Represents a single instance of the DT problem. 
In other words, implements (D, G, C, Q) as a class. 
Implements the random, Ratiocoll, and EpsilonGreedy algorithms. 
Supports integer-valued CSV files only for the time being. 
"""

EPSILON_PROB = 0.000001

class DT:
    def __init__(self, task: AbstractTask, slices: npt.NDArray, explore_scale: float, batch_size: int):
        self.task = task
        self.sources = task.additional_sources
        self.n = len(self.sources)
        self.costs = [source.amortized_cost_per_tuple() for source in self.sources]
        self.slices = slices
        self.m = len(slices)
        self.explore_scale = explore_scale
        self.batch_size = batch_size

    def __repr__(self):
        s = "n: " + str(self.n) + ", m: " + str(self.m) + "\n"
        s += "costs: " + str(self.costs) + "\n"
        s += "slices: " + str(self.slices) + "\n"
        return s

    def run(self, algorithm: str, query_counts: npt.NDArray) -> dict:
        """
        params:
            algorithm: either 'ratiocoll', 'exploreexploit', or 'random' for now
            query_counts: m-length vector denoting minimum query counts
        returns: 
            1. Additional data acquired for each slice, in same order. Encoded
                in raw format.
            2. DT stats dictionary. 
        """
        for source in self.sources:
            source.reset()
        additional_data = pd.DataFrame(columns=self.task.all_column_names())
        remaining_query = np.copy(query_counts)
        total_cost = 0
        # Initialize stat tracker
        if algorithm == "ratiocoll":
            sql_readable_slices = self.task.recode_slice_to_sql_readable(self.slices)
            stat_tracker = [source.slices_count(sql_readable_slices) for source in self.sources]
        elif algorithm == "exploreexploit":
            stat_tracker = np.zeros((self.n, self.m), dtype=int)
            # Compute the number of exploration iterations
            Q = np.sum(query_counts) * self.explore_scale
            explore_iters = math.ceil((0.5 * math.pow(Q, 2 / 3)) / self.batch_size)
            # Round up explore_iters to ensure uniform exploration
            self.explore_iters = math.ceil(explore_iters / self.n) * self.n
        elif algorithm == "random":
            stat_tracker = None
        else:
            raise ValueError("Unsupported algorithm.")
        # Keep track of which sources are chosen
        sources_cnt = [0 for i in range(self.n)]

        itr = 0
        # Loop while query is not satisfied
        while np.any(remaining_query > 0):
            chosen_source = self.choose_ds(algorithm, itr, stat_tracker, remaining_query)
            if chosen_source is None:
                break
            sources_cnt[chosen_source] += 1
            itr += 1
            # Issue query
            query_result, query_cost = self.task.get_additional_data(chosen_source, self.batch_size)
            if query_result is None:
                continue
            # Keep track of stats
            total_cost += self.costs[chosen_source] * len(query_result)
            # Split query result to x, y
            result_x, _ = utils.split_df_xy(query_result, self.task.y_column_name())
            if len(result_x) == 0:
                self.sources[chosen_source].reset()
            # slice_ownership[i][j] denotes whether ith tuple belongs to slice j
            print(result_x)
            slice_ownership = self.task.slice_ownership(result_x, self.slices)
            print("slice_ownership\n", slice_ownership)
            # Count the frequency of each subgroup in query result
            for i in range(len(result_x)):
                if algorithm == "exploreexploit":
                    stat_tracker[chosen_source] += slice_ownership[i]
                remaining_query -= slice_ownership[i]
                result_row = query_result.iloc[i, :].to_frame().T
                if any(slice_ownership[i] & remaining_query > 0):
                    additional_data = pd.concat([additional_data, result_row], ignore_index=True)
        # Generate stats
        ret = {
            "data": additional_data,
            "cost": total_cost,
            "sources": sources_cnt
        }
        return ret

    def choose_ds(self, algorithm: str, itr: int, stat_tracker, remaining_query: npt.NDArray) -> int:
        if algorithm == "random":
            while True:
                ds = rnd.choice(range(self.n))
                if not self.sources[ds].has_next():
                    continue
                return ds
        elif algorithm == "exploreexploit":
            if itr < self.explore_iters:
                priority_source = itr % self.n
                return priority_source
            else:
                P = np.maximum(stat_tracker / max(1, np.sum(stat_tracker)),
                               np.full(np.shape(stat_tracker), EPSILON_PROB))
                c_over_p = np.reshape(self.costs, (self.n, 1)) / P
                min_c_over_p = np.amin(c_over_p, axis=0)
                group_scores = remaining_query * min_c_over_p
                priority_group = np.argmax(group_scores)
                source_scores = c_over_p[:, priority_group]
                sorted_sources = np.argsort(source_scores)
                for source in sorted_sources:
                    if self.sources[source].has_next():
                        return source
        elif algorithm == "ratiocoll":
            P = np.maximum(stat_tracker / max(1, np.sum(stat_tracker)),
                           np.full(np.shape(stat_tracker), EPSILON_PROB))
            c_over_p = np.reshape(self.costs, (self.n, 1)) / P
            min_c_over_p = np.amin(c_over_p, axis=0)
            group_scores = remaining_query * min_c_over_p
            priority_group = np.argmax(group_scores)
            source_scores = c_over_p[:, priority_group]
            sorted_sources = np.argsort(source_scores)
            for source in sorted_sources:
                if self.sources[source].has_next():
                    return source