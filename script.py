print("Hello, Nishant! Your script is running successfully.")


name = input("Enter your name: ")
print(f"Hello, {name}! Welcome to Python programming.")


num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

sum_result = num1 + num2
print(f"The sum of {num1} and {num2} is {sum_result}")

import yfinance as yf

# Enter the stock ticker symbol (e.g., "AAPL" for Apple, "TSLA" for Tesla)
ticker_symbol = input("Enter stock ticker symbol: ")

# Download the stock data
stock_data = yf.download(ticker_symbol, start="2024-01-01", end="2025-02-16")

# Display the first few rows
print(stock_data.head())
