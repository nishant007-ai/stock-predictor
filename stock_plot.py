import yfinance as yf
import matplotlib.pyplot as plt

# Enter stock ticker symbol
ticker_symbol = input("Enter stock ticker symbol: ").strip().upper()

# Download stock data
stock_data = yf.download(ticker_symbol, start="2020-01-01", end="2025-02-16")

# Plot the closing price
plt.figure(figsize=(12,6))
plt.plot(stock_data.index, stock_data["Close"], label=f"{ticker_symbol} Closing Price", color='blue')

# Labels and title
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.title(f"{ticker_symbol} Stock Price Trend")
plt.legend()

# Show the graph
plt.show()

