from sklearn.linear_model import LinearRegression
import pandas as pd

# Sample dataset
data = {
    'Years of Experience': [1, 2, 3, 4, 5],
    'Salary ($)': [35000, 40000, 45000, 50000, 55000]
}
df = pd.DataFrame(data)

# Define features and target
X = df[['Years of Experience']]
y = df['Salary ($)']

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)
print(f"Coefficients: {model.coef_}, Intercept: {model.intercept_}")
print(f"Predicted Salaries: {predictions}")
