import numpy as np

# Task a)
# Implement a method, calculating a base change matrix.
# Input: lists sourceBase and targetBase - lists of vectors (e.g. [np.array([1, 2, 3]), np.array([2, 0, 1]), ...])
# Output: Matrix A - a 2D np.array, with len(sourceBase) x len(targetBase) entries
def changeBase(sourceBase: list, targetBase: list) -> np.array:
  source_matrix = np.array(sourceBase).T
  target_matrix = np.array(targetBase).T
  A = np.linalg.inv(target_matrix) @ source_matrix
  return A

# Task b)
# Implement a method, checking if a subBase spans a Subvectorspace of the space spanned by the given base
# Input: lists sourceBase and subSpace - lists of vectors (e.g. [np.array([1, 2, 3]), np.array([2, 0, 1]), ...])
# Output: bool
def spansSubSpace(base: list, subBase: list) -> bool:
  base_matrix = np.array(base).T
  for v in subBase:
    v = v.reshape(-1)
    x, residuals, rank, s = np.linalg.lstsq(base_matrix, v, rcond=None)
    residual = np.linalg.norm(base_matrix @ x - v)
    if residual > 1e-6:
      return False
  return True
