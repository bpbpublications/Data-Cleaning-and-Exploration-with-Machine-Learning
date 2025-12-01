import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("sample.csv")
y_true = data["y_true"]
y_pred = data["y_pred"]
# Residuals
residuals = y_true - y_pred

# Plot residuals
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()
