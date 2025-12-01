from sklearn.linear_model import LinearRegression
import numpy as np

# Add a numerical day count
data['Day'] = np.arange(len(data))

# Note: This numeric encoding assumes equal spacing between observations (which may not hold in real-world time series)

# Define input (X) and output (y)
X = data[['Day']]
y = data['Close']

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Forecast the next 30 days
future_days = pd.DataFrame({'Day': np.arange(len(data), len(data) + 30)})
forecast = model.predict(future_days)

# Plot original data and forecast
plt.plot(data['Day'], data['Close'], label='Original Data')
plt.plot(future_days['Day'], forecast, label='Forecast', linestyle='--')
plt.title("Linear Regression Forecasting")
plt.xlabel("Day")
plt.ylabel("Closing Price")
plt.legend()
plt.show()
