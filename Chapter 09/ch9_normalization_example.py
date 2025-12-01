from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# Sample dataset
data = {'Income ($)': [50000, 60000, 70000], 'Age (years)': [30, 35, 40]}
df = pd.DataFrame(data)

# Normalize data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df)

# Apply PCA
pca = PCA(n_components=1)
reduced_data = pca.fit_transform(normalized_data)
print(reduced_data)
