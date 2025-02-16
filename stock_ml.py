import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Enter stock ticker symbol
ticker_symbol = input("Enter stock ticker symbol: ").strip().upper()

# Download stock data
stock_data = yf.download(ticker_symbol, start="2020-01-01", end="2025-02-16")

# Create features (X) and labels (y)
stock_data["Tomorrow"] = stock_data["Close"].shift(-1)
stock_data.dropna(inplace=True)

X = stock_data[["Open", "High", "Low", "Close", "Volume"]]
y = stock_data["Tomorrow"]

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future stock prices
predictions = model.predict(X_test)

# Show first 5 predictions
print("📈 Next Day Stock Price Predictions:")
print(predictions[:5])
