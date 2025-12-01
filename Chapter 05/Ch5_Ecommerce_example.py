import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load e-commerce dataset
df = pd.read_csv('ecommerce.csv')
print("Initial data loaded:")
print(df.head())


# Impute missing values
df['avg_spending'].fillna(df['avg_spending'].mean(), inplace=True)
print("\nMissing values in 'avg_spending' filled with mean:")
print(df['avg_spending'].describe())


# Standardize date formats
df['last_purchase'] = pd.to_datetime(df['last_purchase'], format='%Y-%m-%d')
print("\nConverted 'last_purchase' to datetime format:")
print(df['last_purchase'].head())


# Scale numerical features
scaler = StandardScaler()
df[['avg_spending', 'browsing_time']] = scaler.fit_transform(df[['avg_spending', 'browsing_time']])
print("\nScaled 'avg_spending' and 'browsing_time':")
print(df[['avg_spending', 'browsing_time']].head())


# One-hot encode categorical features
encoder = OneHotEncoder()
encoded = pd.DataFrame(encoder.fit_transform(df[['product_category']]).toarray())
df = df.join(encoded)
print("\nFinal preprocessed dataset:")
print(df.head())
