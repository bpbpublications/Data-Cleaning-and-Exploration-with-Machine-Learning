import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
df = pd.read_csv('house_prices.csv')

# Inspect data structure
print(df.info())

# Check for missing values
print(df.isnull().sum())


# Histogram for house Prices
df['Price'].plot(kind='hist', bins=30, title='House Prices Distribution')
plt.show()

# Box plot to detect outliers
df.boxplot(column=['Price'], vert=False)
plt.show()


# Scatter plot for house size vs. Price
sns.scatterplot(data=df, x='Size_sqft', y='Price')

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

#multivariate analysis

sns.pairplot(df[['Price', 'Size_sqft', 'Num_Bedrooms']])
plt.suptitle('Pairplot of Price, Size, and Bedrooms', y=1.02)
plt.show()

