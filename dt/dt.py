import re
import numpy as np
import pandas as pd

"""
Represents a single instance of the DT problem. 
In other words, implements (D, G, C, Q) as a class. 
Implements the random, Ratiocoll, and EpsilonGreedy algorithms. 
"""
class DT:
    def __init__(self, sources, costs, features, stats=None, batch=None):
        """
        params
            sources: a list of CSV filenames
            costs: a list of floating point cost for each source
            features: a list of (feature_name, feature_bins) tuples, where the
                former denotes the name of the feature in the dataset and the
                latter is a collection of values or bins for said feature
            stats: an optional array of dataframes with columns equal to 
                features with fully enumerated subgroups, and an extra
                column named 'count' which denotes the exact or approximate
                count, for each data source
            batch: number of rows to be read in at once
        """
        self.num_sources = len(sources)
        self.num_features = len(features)
        self.sources = sources
        self.costs = np.array(costs)
        self.features = features
        if stats is None:
            
        else:
            self.stats = stats
        self.csv_batch_size = csv_batch_size
    
    def gather_data(self, groups, query_counts):
        """
        params
            groups: a dataframe with columns equal to the features and each 
                row representing a slice, where a column is set to None if it 
                is not used, exact match or binning supported
            queries: a list of requested minimum query counts for each group,
                with groups ordered in the same way as the groups argument
        """