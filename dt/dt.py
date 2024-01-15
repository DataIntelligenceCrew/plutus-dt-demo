import re
import csv
import itertools
import numpy as np
import pandas as pd
import math
import psycopg2
from sklearn.preprocessing import LabelEncoder

class DBSource:
        def __init__(self, host, database, user, password, query, y_name):
                self.host = host
                self.database = database
                self.user = user
                self.password = password
                self.query = query
                self.y_name = y_name

        # Runs the query and returns result as DF
        def get_query_result(self):
                conn = conn = psycopg2.connect(
                        host = self.host, 
                        database = self.database, 
                        user = self.user, 
                        password = self.password
                )
                cur = conn.cursor()
                cur.execute(self.query)
                rows = cur.fetchall()
                col_names = [desc[0] for desc in cur.description]
                df = pd.DataFrame(rows, columns=col_names)
                cur.close()
                conn.close()
                return df

        def __str__(self):
                s = str(self.host) + ' ' + self.database + ' ' + self.user + ' ' + self.query + ' ' + self.y_name
                return s

"""
Represents a single instance of the DT problem. 
In other words, implements (D, G, C, Q) as a class. 
Implements the random, Ratiocoll, and EpsilonGreedy algorithms. 
Supports integer-valued CSV files only for the time being. 
"""
class DT:
    def __init__(self, sources=None, costs=None, slices=None, stats=None, batch=1000):
        """
        params
            sources: a list of CSV filenames
            costs: a list of floating point cost for each source
            slices: a list of slices, where each slice is a list of strings formatted
                            as '[min, max]' (inclusive) or '(min, max)' (exclusive). 
                            inclusive and exclusive brackets can be combined. 
                            min, max should be parseable as float
                            the elements in the slices should correspond to the features
                            in the features argument, in the same order
            stats: an optional list of numpy vectors that count the number of
                            each slice, in the order given in the slices argument
                            format is stats[source][slice]
            batch: number of rows to be read in at once
        """
        self.n = len(sources) # number of sources
        if type(sources[0]) == str: # Source is defined as CSV filename
            self.sources = sources # data sources & file readers
            self.readers = [CSVChunkReader(filename, batch) for filename in sources]
        else: # Source is defined as DBSource
            self.sources = sources
        self.costs = np.array(costs) # costs
        # Parsing slices
        if type(slices[0][0] == tuple):
            self.slices = slices
        else:
            self.slices = parse_slices(slices)
        self.m = len(slices)
        # Stat tracker (initialize if not given)
        if stats is None:
            self.stats = np.zeros((self.n, self.m))
            self.priors = False
        else:
            self.stats = stats
            self.priors = True
        self.batch = batch # batch size

    def __str__(self):
        s =  "n: " + str(self.n) + "\n"
        s += "sources: " + str(self.sources) + "\n"
        s += "costs: " + str(self.costs) + "\n"
        s += "slices: " + str(self.slices) + "\n"
        s += "stats: " + str(self.stats) + "\n"
        s += "batch: " + str(self.batch) + "\n"
        return s
        
    def run(self, query_counts):
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
        # Known setting, use RatioColl
        if self.priors:
                return self.ratiocoll(query_counts)
        else:
                return self.exploreexploit(query_counts)
    
    def exploreexploit(self, query_counts):
        Q = sum(query_counts)
        unified_set = None
        unified_ys = []
        remaining_query = np.copy(query_counts)

        explore_iters = max(int(0.5 * Q / self.batch) + 1, self.n)
        
        i = 0
        while np.any(remaining_query > 0):
            if i < explore_iters:
                priority_source = i % self.n
            else:
                P = np.maximum(self.stats / 4200, 0.1)
                C_over_P = np.reshape(self.costs.T, (self.n, 1)) / P
                min_C_over_P = np.amin(C_over_P, axis=0)
                group_scores = remaining_query * min_C_over_P
                priority_group = np.argmax(group_scores)
                priority_source = np.argmin(C_over_P[:,priority_group], axis=0)
            if type(self.sources[0]) == str: # Sources are CSV files
                query_result = np.array(self.readers[priority_source].next())
            else: # Sources are DB queries
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

def belongs_to_slice(slice_, result_row):
    """
    returns whether result_row belongs to slice_
    result_row is assumed to be a dataframe w/ feature titles
    """
    for i in range(len(result_row)):
        xi = result_row[i]
        if xi < slice_[i][0]:
            return False
        if xi > slice_[i][1]:
            return False
    return True

def split_xy(df, y_name):
    df_y = df[y_name]
    df_x = df.drop(y_name, axis=1)
    return df_x, df_y

def int_encode(df, cols):
    label_encoder = LabelEncoder()
    for col in cols:
        if col in df.columns:
            df[col] = label_encoder.fit_transform(df[col])
    return df

def process_df(df):
    df = pd.get_dummies(df,columns=onehot_cols)
    df = int_encode(df, int_cols)
    return df

# These columns will be one-hot encoded
onehot_cols = [
]
# These columns will be encoded as integers
int_cols = [
    'carrier_mkt',
    'carrier_op',
    'origin_state_abr',
    'dest_state'
]

def parse_slices(slices):
    """
    Parse a slice formatted using string into a slice formatted using tuple. 
    params:
        slices: a list of slices, where each slice is a list of strings formatted
                        as '(min, max)' for each feature. min, max should be parseable as
                        float. 
    returns:
        slices reformatted using tuples of floats instead
    """
    reformatted_slices = []
    for slice_ in slices:
        reformatted_slice = []
        for feature in slice_:
            middle_str = feature[1:-1]
            split_middle = middle_str.split(',')
            feature = (float(split_middle[0]), float(split_middle[1]))
            reformatted_slice.append(feature)
        reformatted_slices.append(reformatted_slice)
    return reformatted_slices
