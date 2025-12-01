from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


data = pd.read_csv('retail_sales_forecasting.csv')
y_pred = data['Predicted_Sales']
y_true = data['Actual_Sales']
# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
r_squared = r2_score(y_true, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r_squared}")
