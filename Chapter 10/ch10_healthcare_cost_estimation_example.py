from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
# Load the dataset
data = pd.read_csv('healthcare_costs.csv')  # Features: 'age', 'bmi', 'procedure_code'; Target: 'cost'

# Define features (X) and target variable (y)
X = data[['age', 'bmi', 'procedure_code']]
y = data['cost']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the random forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error (MSE): {mse:.2f}")
