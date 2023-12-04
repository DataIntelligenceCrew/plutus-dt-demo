import re
import csv
import itertools
import numpy as np
import pandas as pd
import math

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
		#self.d = len(features) # number of featurs
		self.sources = sources # data sources & file readers
		self.readers = [CSVChunkReader(filename, batch) for filename in sources]	
		self.costs = np.array(costs) # costs
		# Parsing slices
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
		unified_set = []
		remaining_query = np.copy(query_counts)

		explore_iters = max((int(math.pow(Q, 2/3)) / 100) + 1, self.n)
		
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
			query_result = np.array(self.readers[priority_source].next())
			# Count the frequency of each subgroup in query result
			for b, result_row in enumerate(query_result):
				slices = self.slice_ownership(result_row)
				self.stats[priority_source][slices] += 1
				if len(slices) > 0:
					unified_set.append(result_row)
				remaining_query[slices] -= 1
				remaining_query = np.maximum(remaining_query, 0)
			i += 1
		unified_set = np.array(unified_set)
		#print(unified_set, len(unified_set), i)
		return unified_set
		
	def ratiocoll(self, query_counts):
		"""
		params:
			query_coutns: (1, m) ndarray denoting query counts for each group
			batch: int, batch size for reading sources
			discard: whether to keep or discard excess tuples
		"""
		# The actual collected set
		unified_set = []
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
			query_result = np.array(self.readers[priority_source].next())
			# Count the frequency of each subgroup in query result
			for b, result_row in enumerate(query_result):
				slices = self.slice_ownership(result_row)
				if len(slices) > 0:
					unified_set.append(result_row)
				remaining_query[slices] -= 1
				remaining_query = np.maximum(remaining_query, 0)
		unified_set = np.array(unified_set)
		#print(unified_set, len(unified_set), query_times)
		return unified_set
	
	def slice_ownership(self, result_row):
		"""
		returns a boolean list denoting the slices the other result_row belongs to
		"""
		ownership = []
		for slice_ in self.slices:
			ownership.append(belongs_to_slice(slice_, result_row))
		return ownership

def belongs_to_slice(slice_, result_row):
	"""
	returns whether result_row belongs to slice_
	"""
	for i in range(len(slice_)):
		if result_row[i] < slice_[i][0]:
			return False
		if result_row[i] > slice_[i][1]:
			return False
	return True


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

# Random CSV files will each have features a, b, c
# a = 0 or 1
# b = 0 to 2
# c = 0 to 3
# Subgroup stats: 
#   000   001   002   003   010   011   012   013   020   021   022   023   100   101   102   103   110   111   112   113   120   121   122   123
# -----------------------------------------------------------------------------------------------------------------------------------------------
# 0 2634  3970  5882  0     873   1308  1966   0,   3419  5230  7893  0     5283  7805  11878 0     1798  2623  4013  0     7001  10351 16073 0
# 1 137   2211  147   255   739   11082 729    1429 901   13247 870   1763  315   4438  289   587   1412  21901 1461  2873  1769  26300 1694  3451
# 2 3291  2230  964   960   3687  2531  1122   1020 456   288   125   151   15602 11134 4830  4671  18163 12840 5348  5430  2281  1498  696   682
# 3 4545  4545  4497  872   1122  1154  1164   241  3398  3304  3356  630   11114 11119 11229 2273  2801  2800  2815  549   8154  8193  8428  1697
# 4 1281  604   856   366   7569  3705  5316   2323 6347  3032  4267  1868  2102  1062  1499  633   12539 6151  8824  3698  10479 5249  7159  3071

if __name__ == '__main__':

		sources = [
				'data/random_csv_0.csv',
				'data/random_csv_1.csv',
				'data/random_csv_2.csv',
				'data/random_csv_3.csv',
				'data/random_csv_4.csv'
		]

		costs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

		# Slices used for random CSV test case:
		# 11X
		# 011
		# 111
		slices = [
			['(0.5,inf)', '(0.5,1.5)', '(-inf,inf)'],
			['(-inf,0.5)', '(0.5,1.5)', '(0.5,1.5)'],
			['(0.5,inf)', '(0.5,1.5)', '(0.5,1.5)']
		]

		#   11X   023   111
		# 0 8434  0     2623
		# 1 27647 1763  21901
		# 2 41781 151   12840
		# 3 8965  630   2800
		# 4 31212 1868  6151
		stats = [
			[8434, 1308, 2623],
			[27647, 11082, 21901],
			[41781, 2531, 12840],
			[8965, 1154, 2800],
			[31212, 3705, 6151],
		]
		
		dt = DT(sources, costs, slices, None, batch=100)
		print(dt)
		
		query_counts = [
			1000,
			1000,
			1000
		]

		dt.run(query_counts)