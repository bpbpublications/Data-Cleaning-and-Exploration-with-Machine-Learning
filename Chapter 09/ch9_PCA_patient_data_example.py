from sklearn.decomposition import PCA
import pandas as pd

# Load patient data
data = pd.read_csv("patient_data.csv")
pca = PCA(n_components=3)
principal_components = pca.fit_transform(data)

print("Explained Variance Ratios:", pca.explained_variance_ratio_)
