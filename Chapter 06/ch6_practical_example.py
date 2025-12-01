import pandas as pd
import matplotlib.pyplot as plt

# Sample dataset
data = {'Income': [30000, 35000, 40000, 42000, 45000, 50000, 55000, 60000, 65000, 100000]}
df = pd.DataFrame(data)

# Descriptive statistics
print(df['Income'].describe())

# Histogram for income distribution
plt.hist(df['Income'], bins=20, edgecolor='black', color='purple')
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

# Box plot for detecting outliers
plt.boxplot(df['Income'], patch_artist=True, boxprops=dict(facecolor='orange'))
plt.title('Income Distribution - Box Plot')
plt.ylabel('Income')
plt.show()
