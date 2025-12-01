import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression

# Simulated data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1.5, 3.8, 8.5, 17.2, 25.3])

# Transform features into polynomial terms (degree=2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Predictions
poly_predictions = poly_model.predict(X_poly)

# Visualization
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, poly_predictions, color='red', label='Polynomial Fit (Degree 2)')
plt.title('Polynomial Regression Example')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
