import re
import csv
import itertools
import numpy as np
import pandas as pd

class CSVChunkReader:
    def __init__(self, filename, chunksize=1000):
        self.filename = filename
        self.chunksize = chunksize
        self.reader = pd.read_csv(filename, chunksize=chunksize)
    
    def next(self):
        try:
            chunk = next(self.reader)
            return chunk
        except StopIteration:
            self.reader = None
            return None

"""
Represents a single instance of the DT problem. 
In other words, implements (D, G, C, Q) as a class. 
Implements the random, Ratiocoll, and EpsilonGreedy algorithms. 
Supports integer-valued CSV files only for the time being. 
"""
class DT:
    def __init__(self, sources, costs, features, stats=None, batch=1000):
        """
        params
            sources: a list of CSV filenames
            costs: a list of floating point cost for each source
            features: a list of (feature_name, min, max) tuples, where the
                former denotes the name of the feature in the dataset and the
                latter is the minimum and maximum value in that feature
                these features are used for subgroup stat tracking
            stats: an optional list of numpy vectors that count the number of
                each lowest-level subgroup ordered in the same way as features
            batch: number of rows to be read in at once
        """
        # n
        self.num_sources = len(sources)
        # d
        self.num_features = len(features)
        # 2^d (if binary)
        self.num_subgroups = 1
        for feature in features:
            feature_range = feature[2] - feature[1] + 1
            self.num_subgroups *= feature_range
        # batch size
        self.batch = batch
        # D
        self.sources = sources
        self.readers = [CSVChunkReader(filename, batch) for filename in sources]
        # C
        self.costs = np.array(costs)
        # dims
        self.features = features
        # Stat tracker
        if stats is None:
            # In unknown setting, keep track of count for each subgroup
            self.stats = [np.zeros(self.num_subgroups, dtype=int)
                          for _ in range(self.num_sources)]
            self.priors = False
            self.N = np.zeros(self.num_sources, dtype=int)
        else:
            # In known setting, just use the provided stats vector
            self.stats = stats
            self.priors = True
            self.stats_N = [ np.sum(self.stats[i]) for i in range(self.num_sources) ]
        # Initialze the subgroup matrix
        # This is a (2^d, d) matrix where each row represents a subgroup
        # and the columns represent features
        # i.e. a row of [0, 1, 0] represents the subgroup which has values 0
        # for feature 0, 1 for feature 1, and 0 for feature 2
        # Example:
        # [[0 0]
        #  [0 1]
        #  [1 0]
        #  [1 1]]
        combinations = []
        for feature in features:
            combinations.append(list(range(feature[1], feature[2] + 1)))
        all_combinations = list(itertools.product(*combinations))
        self.subgroups = np.array(all_combinations)
        # Initialize the slice to subgroup ID dictionary
        self.subgroup_to_id_dict = dict()
        for i, subgroup in enumerate(all_combinations):
            self.subgroup_to_id_dict[subgroup] = i
    
    def __str__(self):
        s =  "n: " + str(self.num_sources) + "\n"
        s += "d: " + str(self.num_features) + "\n"
        s += "subgroup count: " + str(self.num_subgroups) + "\n"
        s += "batch: " + str(self.batch) + "\n"
        s += "sources: " + str(self.sources) + "\n"
        s += "costs: " + str(self.costs) + "\n"
        s += "features: " + str(self.features) + "\n"
        s += "stats: " + str(self.stats) + "\n"
        s += "subgroups: " + str(self.subgroups) + "\n"
        s += "dict: " + str(self.subgroup_to_id_dict)
        return s
    
    def run(self, patterns, query_counts):
        """
        params
            patterns: ndarray with shape (m, d) where each row is a group of
                interest, totalling m groups, and each column is the value that
                the group should take in the d^th dimension, with dimensions
                ordered in the same order as self.features
                dimensions that do not matter should have a negative value
                i.e. pattern 1X0 is encoded as row [1, -1, 0]
            query_counts: vector with length m denoting query count for each
                group requested
        """
        # First, transform the (m, d) pattern matrix to (m, 2^d) subgroup
        # inclusion matrix where each row is a group, and each column is a
        # subgroup ID in increasing order
        pattern_to_subgroup = []
        for pattern in patterns:
            # Remove the X features in this pattern
            x_indices = np.where(pattern < 0)[0] # X dimension indices
            no_x_subgroups = np.delete(self.subgroups, x_indices, axis=1)
            no_x_pattern = np.delete(pattern, x_indices)
            # Subtract pattern from subgroup
            diff = no_x_subgroups - no_x_pattern
            subgroup_incl = np.all(diff == 0, axis=1).astype(int)
            pattern_to_subgroup.append(subgroup_incl)
        self.subgroup_incl = np.array(pattern_to_subgroup)

        # Known setting, use RatioColl
        if self.priors:
            ds_index = self.ratiocoll(query_counts, self.batch)
        else:
            pass
    
    def ratiocoll(self, query_counts, batch):
        """
        params:
            subgroup_incl: (m, 2^d) ndarray denoting the features in each group
            query_coutns: (1, m) ndarray denoting query counts for each group
            batch: int, batch size for reading sources
            discard: whether to keep or discard excess tuples
        """
        m = len(query_counts)
        # P is the n by m matrix of probability of each group in each source
        P = []
        # Each row of P_ij is computed by matmul subgroup_incl and subgroup_cnt
        for source_stat in self.stats:
            prod = self.subgroup_incl @ source_stat.T / np.sum(source_stat)
            P.append(prod)
        P = np.array(P)
        print(P)

        # The actual collected set
        unified_set = []
        remaining_query = np.copy(query_counts)

        # Precompute some matrices
        # (n, m) matrix, where P has been normalized by C
        C_over_P = np.reshape(self.costs.T, (self.num_sources,1)) / P
        # (1, m) matrix, where we find the minimum C/P for each group
        min_C_over_P = np.amin(C_over_P, axis=0)

        query_times = 0
        while np.any(remaining_query > 0):
            query_times += 1
            # Score for each group, (1, m)
            group_scores = remaining_query * min_C_over_P
            print("group scores:", group_scores)
            # Priority group & source
            priority_group = np.argmax(group_scores)
            print("priority group:", priority_group)
            priority_source = np.argmin(C_over_P[:,priority_group], axis=0)
            print("priority source:", priority_source)
            # Batch query chosen source
            query_result = np.array(self.readers[priority_source].next())
            #print(query_result)
            # Count the frequency of each subgroup in query result
            subgroup_cnts = np.zeros(self.num_subgroups, dtype=int)
            subgroup_indices = np.empty(self.batch)
            for b, result_row in enumerate(query_result):
                subgroup = self.subgroup_to_id_dict[tuple(result_row)]
                # Binary indicator of the patterns that this tuple satisfies
                result_is_pattern = np.equal(self.subgroup_incl[:,subgroup], [1])
                # Binary indicator of the patterns which still have remaining query
                remaining_groups = np.greater(remaining_query, [0])
                # All remaining groups for which this tuple satisfies
                reduces_group_query = np.logical_and(result_is_pattern, remaining_groups).astype(int)
                #print(result_row, subgroup, result_is_pattern, remaining_groups, reduces_group_query, remaining_query)
                # Bookkeeping
                if np.sum(reduces_group_query > 0):
                    unified_set.append(result_row)
                    remaining_query -= reduces_group_query
        unified_set = np.array(unified_set)
        print(unified_set, len(unified_set), query_times)
        return np.array(unified_set)

if __name__ == '__main__':
    sources = [
        'random_csv_0.csv',
        'random_csv_1.csv',
        'random_csv_2.csv',
        'random_csv_3.csv',
        'random_csv_4.csv'
    ]
    costs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    features = [("a", 0, 1), ("b", 0, 2), ("c", 0, 3)]
    stats = [
        [2634, 3970, 5882, 0, 873, 1308, 1966, 0, 3419, 5230, 7893, 0, 5283, 7805, 11878, 0, 1798, 2623, 4013, 0, 7001, 10351, 16073, 0],
        [137, 2211, 147, 255, 739, 11082, 729, 1429, 901, 13247, 870, 1763, 315, 4438, 289, 587, 1412, 21901, 1461, 2873, 1769, 26300, 1694, 3451],
        [3291, 2230, 964, 960, 3687, 2531, 1122, 1020, 456, 288, 125, 151, 15602, 11134, 4830, 4671, 18163, 12840, 5348, 5430, 2281, 1498, 696, 682],
        [4545, 4545, 4497, 872, 1122, 1154, 1164, 241, 3398, 3304, 3356, 630, 11114, 11119, 11229, 2273, 2801, 2800, 2815, 549, 8154, 8193, 8428, 1697],
        [1281, 604, 856, 366, 7569, 3705, 5316, 2323, 6347, 3032, 4267, 1868, 2102, 1062, 1499, 633, 12539, 6151, 8824, 3698, 10479, 5249, 7159, 3071]
    ]
    stats = np.array(stats)
    
    dt = DT(sources, costs, features, stats, batch=100)
    print(dt)

    patterns = [
        [0, 1, -1],
        [-1, 1, 2]
    ]
    patterns = np.array(patterns)
    query_counts = np.array([2000, 1000])

    dt.run(patterns, query_counts)