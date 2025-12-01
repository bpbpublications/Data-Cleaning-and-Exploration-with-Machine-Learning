import numpy as np
from scipy.linalg import svd

# Example matrix
matrix = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Perform SVD
U, S, Vt = svd(matrix)

print("U Matrix:\n", U)
print("Singular Values:\n", S)
print("V^T Matrix:\n", Vt)
