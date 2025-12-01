from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
data = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])

# Apply K-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)

# Visualize clusters
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', label='Data Points')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x', label='Centroids')
plt.title("K-Means Clustering")
plt.legend()
plt.show()
