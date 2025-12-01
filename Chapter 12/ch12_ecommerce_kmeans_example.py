from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Example dataset
data = {
    'Purchase Frequency': [5, 20, 15, 8, 50],
    'Average Order Value': [100, 200, 150, 120, 500],
    'Browsing Duration (minutes)': [15, 60, 45, 25, 120]
}
df = pd.DataFrame(data)

# Normalize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Determine optimal clusters using the elbow method
inertia = []
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 6), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Apply K-means with optimal clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)
print(df)
