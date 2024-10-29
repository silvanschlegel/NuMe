import numpy as np

# Task a)
# Implement a method, calculating the LU factorization of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: Matrices P, L and U - same shape as A each.
def lu(A):
    n = A.shape[0]
    U = A.copy()
    L = np.zeros_like(A)
    P = np.eye(n)

    for k in range(n):
        pivot = np.argmax(np.abs(U[k:, k])) + k

        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        L[k, k] = 1.0
        if U[k, k] != 0:
            for i in range(k + 1, n):
                L[i, k] = U[i, k] / U[k, k]
                U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
                U[i, k] = 0.0  # Ensure lower entries are zero
        else:
            L[k + 1:, k] = 0.0
    return P, L, U

# Task b)
# Implement a method, calculating the determinant of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: The determinant - a floating number
def determinant(A):
    try:
        P, L, U = lu(A)
        n = A.shape[0]

        perm = np.argmax(P, axis=1)

        def permutation_parity(perm):
            n = len(perm)
            visited = [False] * n
            parity = 0
            for i in range(n):
                if not visited[i]:
                    j = i
                    cycle_size = 0
                    while not visited[j]:
                        visited[j] = True
                        j = perm[j]
                        cycle_size += 1
                    parity += cycle_size - 1
            return (-1) ** parity

        det_P = permutation_parity(perm)

        det_U = np.prod(np.diag(U))

        det_A = det_P * det_U
        return det_A
    except ValueError:
        #singular matrix
        return 0.0