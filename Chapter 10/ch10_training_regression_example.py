# Advanced dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Advanced dataset
data = {
    'Size (sq. ft.)': [1500, 2000, 2500, 3000, 3500],
    'Bedrooms': [3, 3, 4, 4, 5],
    'Location Score': [7, 8, 9, 9, 10],
    'Price ($)': [300000, 400000, 500000, 600000, 700000]
}
df = pd.DataFrame(data)

# Define features and target
X = df[['Size (sq. ft.)', 'Bedrooms', 'Location Score']]
y = df['Price ($)']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# --- Model metrics ---
# R² score
r2 = r2_score(y, y_pred)

# Adjusted R² score
n = X.shape[0]   # number of observations
p = X.shape[1]   # number of predictors
adj_r2 = 1 - (1-r2) * (n-1) / (n-p-1)

# RMSE
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Print outputs
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")
print(f"RMSE: {rmse:.2f}")

# --- Residual analysis ---
residuals = y - y_pred

plt.scatter(y_pred, residuals, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Prices ($)")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.show()
