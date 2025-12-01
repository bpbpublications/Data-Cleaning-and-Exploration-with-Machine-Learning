import seaborn as sns
import matplotlib.pyplot as plt

# Sample dataset
import pandas as pd
data = {'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
        'variable2': [10, 15, 7, 12, 17, 9, 14, 19, 11]}
df = pd.DataFrame(data)

# Create a Seaborn box plot
sns.boxplot(x='category', y='variable2', data=df, hue='category' , palette='coolwarm')

# Customize with Matplotlib
plt.title('Enhanced Box Plot with Seaborn and Matplotlib', fontsize=16)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Variable 2', fontsize=12)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.show()
