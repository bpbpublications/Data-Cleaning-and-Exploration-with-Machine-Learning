from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
data = np.array([
    [1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13],
    [5, 6], [6, 7], [25, 30], [26, 31]
])

# Apply DBSCAN
dbscan = DBSCAN(eps=2.5, min_samples=2)
dbscan.fit(data)

# Visualize clusters
plt.scatter(data[:, 0], data[:, 1], c=dbscan.labels_, cmap='viridis', s=100)
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Cluster Label")
plt.grid()
plt.show()
