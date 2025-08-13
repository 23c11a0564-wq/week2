import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Download stock data (e.g., Apple)
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2023-12-31")

# 2. Use only the 'Close' column
data = data[["Close"]]
data["Prediction"] = data[["Close"]].shift(-30)  # Predict 30 days into the future

# 3. Prepare the dataset
X = np.array(data.drop(["Prediction"], axis=1))[:-30]
y = np.array(data["Prediction"])[:-30]

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# 6. Testing
predictions = lr.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)

# 7. Forecast future 30 days
X_future = data.drop(["Prediction"], axis=1)[-30:]
forecast = lr.predict(X_future)

# 8. Plot the results
plt.figure(figsize=(12, 6))
plt.title("Stock Price Prediction - Linear Regression")
plt.plot(data["Close"], label="Actual Price")
plt.plot(range(len(data)-30, len(data)), forecast, label="Predicted Price", color="red")
plt.xlabel("Days")
plt.ylabel("Close Price USD")
plt.legend()
plt.show()
