import numpy as np

# Task a)
# Implement the gaussian elimination method, to solve the given system of linear equations;
# Add partial pivoting to increase accuracy and stability of the solution;
# Return the solution for x
# Assume a square matrix
def solveLinearSystem(A, b):

  A = A.copy().astype(float)
  b = b.copy().astype(float)

  n = len(b)

  x = np.ones(b.shape)

  # Gaussian elimination
  for k in range(n):
    max_row_index = np.argmax(np.abs(A[k:, k])) + k
    max_pivot = A[max_row_index, k]

    if max_row_index != k:
      A[[k, max_row_index], :] = A[[max_row_index, k], :]
      b[[k, max_row_index]] = b[[max_row_index, k]]

    pivot = A[k, k]
    if np.isclose(pivot, 0):
      if np.allclose(A[k, :], 0) and np.isclose(b[k], 0):
        continue
      else:
        # Inconsistent
        raise ValueError("System is inconsistent or matrix is singular.")

    # Eliminate entries below the pivot
    for i in range(k + 1, n):
      factor = A[i, k] / pivot
      A[i, k:] -= factor * A[k, k:]
      b[i] -= factor * b[k]

  # backwards substitution
  for i in range(n - 1, -1, -1):
    if np.allclose(A[i, :], 0):
      if np.isclose(b[i], 0):
        # Free variable
        x[i] = 0
      else:
        # Inconsistent
        raise ValueError("System is inconsistent or matrix is singular.")
    else:
      x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

  result = np.ones(b.shape)
  result[:] = x

  return result


# Task b)
# Implement a method, checking whether the system is consistent or not;
# Obviously, you're not allowed to use any method solving that problem for you.
# Return either true or false
def isConsistent(A,b):
  A = A.copy().astype(float)
  b = b.copy().astype(float)

  n, m = A.shape

  Ab = np.hstack((A, b.reshape(-1, 1)))

  # Gaussian elimination
  for k in range(min(n, m)):
    max_row_index = np.argmax(np.abs(Ab[k:, k])) + k
    max_pivot = Ab[max_row_index, k]

    if max_row_index != k:
      Ab[[k, max_row_index], :] = Ab[[max_row_index, k], :]

    pivot = Ab[k, k]
    if np.isclose(pivot, 0):
      # pivot is zero, cant eliminate in this column
      continue

    # eliminate entries below the pivot
    for i in range(k + 1, n):
      factor = Ab[i, k] / pivot
      Ab[i, k:] -= factor * Ab[k, k:]

  # inconsistenciey check
  for i in range(n):
    # If all coefficients are zero in a row
    if np.allclose(Ab[i, :-1], 0):
      # If the constant term is not zero, the system is inconsistent
      if not np.isclose(Ab[i, -1], 0):
        return False  # Inconsistent system

  return True
