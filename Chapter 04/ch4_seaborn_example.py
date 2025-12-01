import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = np.random.rand(10, 12)
sns.heatmap(data, annot=True, cmap='coolwarm')
plt.title('Heatmap')
plt.show()

# Use a known dataset to avoid confusion
df = sns.load_dataset("iris")

sns.pairplot(df)
plt.suptitle('Pairplot of Iris Dataset', y=1.02)  # Adjust title position
plt.show()

sns.pairplot(df.sample(50))

#styling

sns.set_theme(style='darkgrid')  # Stays active until explicitly changed
sns.lineplot(x=[1, 2, 3], y=[4, 5, 6])
plt.title('Line Plot with Darkgrid Theme')
plt.show()

#color palette

sns.set_palette('pastel')
sns.barplot(x=['A', 'B', 'C'], y=[3, 7, 5])
plt.title('Bar Plot with Pastel Palette')
plt.show()
