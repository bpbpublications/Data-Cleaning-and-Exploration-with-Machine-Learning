from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
data = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])

# Perform hierarchical clustering using Ward’s method
linked = linkage(data, method='ward')

# Plot dendrogram
plt.figure(figsize=(8, 4))
dendrogram(linked, labels=['A', 'B', 'C', 'D', 'E', 'F'], leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
