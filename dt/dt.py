import re
import itertools
import numpy as np
import pandas as pd

"""
Represents a single instance of the DT problem. 
In other words, implements (D, G, C, Q) as a class. 
Implements the random, Ratiocoll, and EpsilonGreedy algorithms. 
Supports integer-valued CSV files only for the time being. 
"""
class DT:
    def __init__(self, sources, costs, features, stats=None):
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
            num_subgroups *= feature_range
        # D
        self.sources = sources
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
        else:
            # In known setting, just use the provided stats vector
            self.stats = stats
            self.priors = True
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
        for i, subgroup in all_combinations:
            self.subgroup_to_id_dict[subgroup] = i
    
    def run(self, patterns, query_counts, batch):
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
            x_indices = np.where(pattern < 0, pattern)[0] # X dimension indices
            no_x_subgroups = np.delete(self.subgroups, negative_indices, axis=1)
            no_x_pattern = np.delete(pattern, negative_indices)
            # Subtract pattern from subgroup
            diff = no_x_subgroups - no_x_pattern
            subgroup_incl = np.all(diff == 0, axis=1).astype(int)
            pattern_to_subgroup.append(subgroup_incl)
        subgroup_incl = np.array(pattern_to_subgroup)

        # Known setting, use RatioColl
        if self.priors:
            return self.ratiocoll(subgroup_incl, query_counts, batch)
        else:
            pass
    
    def ratiocoll(self, subgroup_incl, query_counts, batch):
        # P is the n by m matrix of probability of each group in each source
        P = []
        # Each row of P_ij is computed by matmul subgroup_incl and subgroup_cnt
        for source_stat in self.stats:
            prod = subgroup_incl @ source_stat.T
            prod /= sum(prod)
            P.append(prod)
        P = np.array(P)

        # (n, m) matrix, where P has been normalized by C
        C_over_P = self.costs.T / P

        