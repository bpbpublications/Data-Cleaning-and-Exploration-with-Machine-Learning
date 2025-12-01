from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

data_scaled = pd.read_csv("sample_data.csv")
# Assuming 'data_scaled' is your standardized input data
tsne = TSNE(n_components=2, perplexity=4, learning_rate=200, random_state=42)
data_tsne = tsne.fit_transform(data_scaled)

# Visualize the transformed data
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=data_scaled['label'], cmap='viridis', alpha=0.7)
plt.title("t-SNE Visualization of High-Dimensional Data")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(label="Class Label")
plt.show()
