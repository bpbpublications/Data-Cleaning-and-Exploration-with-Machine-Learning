from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('marketing_data.csv')
features = ['age', 'income', 'spending_score', 'purchase_frequency']

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Apply PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Visualize the reduced data
plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.7)
plt.title("PCA: Reduced Dimensionality")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
