from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

# Load and prepare customer dataset
customer_data = pd.read_csv("customer_data.csv")

# Apply t-SNE with 2D output
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
reduced_data = tsne.fit_transform(customer_data)

# Plot the results
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6)
plt.title("Customer Behavior Clusters")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.show()
