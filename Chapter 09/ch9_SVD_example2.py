from scipy.linalg import svd
import numpy as np

# Simulated financial dataset: 200 days of returns for 50 assets
data = np.random.rand(200, 50)

# Perform Singular Value Decomposition
U, Sigma, Vt = svd(data)

# Display the top 5 singular values to understand key variance contributors
print("Top Singular Values:", Sigma[:5])
