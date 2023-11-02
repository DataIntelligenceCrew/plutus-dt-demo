import numpy as np


def match_pattern(ndarray, pattern):
  """Matches a pattern to a NumPy ndarray.

  Args:
    ndarray: A NumPy ndarray with m columns, where each column represents a
      feature and each row represents a possible combination of features.
    pattern: A NumPy vector with m elements, where any nonnegative number means
      that that feature must be exactly matched, and a negative number means
      that feature is skipped.

  Returns:
    A NumPy vector with T/F values indicating whether each row in the ndarray
    matches the pattern.
  """

  # Create a mask for the pattern.
  mask = np.ones(ndarray.shape[1], dtype=bool)
  for i in range(len(pattern)):
    if pattern[i] < 0:
      mask[i] = False
  print(mask)

  # Match the pattern to the ndarray.
  matches = np.all(ndarray[:, mask] == pattern[mask], axis=1)

  return matches


# Example usage:

ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
pattern = np.array([0, -1])

matches = match_pattern(ndarray, pattern)

print(matches)
