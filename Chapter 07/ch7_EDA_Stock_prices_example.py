import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('ecommerce_data.csv')

# Convert date column to datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Show first few rows to preview the data
print("Data preview before visualization:")
print(df.head())

# Plot daily closing prices
df['close'].plot(title='Daily Closing Prices', figsize=(10, 6))
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

# Fill missing closing prices using forward fill
df['close'] = df['close'].fillna(method='ffill')

# Calculate daily returns
df['daily_return'] = df['close'].pct_change()

# Show first few rows to verify
print("Data preview with daily returns:")
print(df[['close', 'volume', 'daily_return']].head())

# Heatmap of daily returns vs volume
sns.heatmap(df[['daily_return', 'volume']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation: Daily Return vs Volume')
plt.show()

# Heatmap of closing price vs volume
sns.heatmap(df[['close', 'volume']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation: Closing Price vs Volume')
plt.show()

# Boxplot to visualize volume outliers
sns.boxplot(x=df['volume'])
plt.title('Trading Volume Outliers')
plt.xlabel('Volume')
plt.show()
