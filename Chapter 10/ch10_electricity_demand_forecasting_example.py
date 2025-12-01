import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
# Load the dataset
data = pd.read_csv('electricity_demand.csv')  # Features: 'hour_of_day', 'temperature'; Target: 'demand'

# Define features (X) and target variable (y)
X = data[['hour_of_day', 'temperature']]
y = data['demand']

# Transform features into polynomial terms
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

# Train the polynomial regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
