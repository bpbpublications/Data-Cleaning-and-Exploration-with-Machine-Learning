import pandas as pd
import numpy as np #added this
from sklearn.impute import SimpleImputer

# Create a housing dataset with missing values
housing_data = pd.DataFrame({
    'Price': [200000, 250000, None, 300000, 400000],
    'Square_Footage': [1500, 1800, np.nan, 2200, 3000]
})

# Initialize median imputer. SimpleImputer treats np.nan as the missing value marker,not Python’s None.

imputer = SimpleImputer(strategy='median')

# Impute missing values for both columns
housing_data['Price'] = imputer.fit_transform(housing_data[['Price']])
housing_data['Square_Footage'] = imputer.fit_transform(housing_data[['Square_Footage']])

print(housing_data)
