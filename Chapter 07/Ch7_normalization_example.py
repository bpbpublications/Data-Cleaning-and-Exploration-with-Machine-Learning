import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Data
data = {'Income': [50000, 70000, 100000], 'Age': [25, 45, 35]}
df = pd.DataFrame(data)

# Normalize
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)

# Convert back to DataFrame with column names
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

print(normalized_df)
