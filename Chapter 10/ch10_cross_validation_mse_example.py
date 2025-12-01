from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import pandas as pd


data = pd.read_csv("sample3.csv")
X = data[["Size_sq_ft","Bedrooms","Location_Score"]]
y = data["Price"]
model = LinearRegression()
cv_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
mean_cv_score = -cv_scores.mean()
print(f"Cross-validated MSE: {mean_cv_score}")
