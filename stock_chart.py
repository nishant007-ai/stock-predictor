import yfinance as yf
import matplotlib.pyplot as plt

# Enter the stock ticker symbol
ticker_symbol = input("Enter stock ticker symbol: ").strip().upper()

# Download stock data
stock_data = yf.download(ticker_symbol, start="2024-01-01", end="2025-02-16")

# Calculate 20-day and 50-day Simple Moving Averages (SMA)
stock_data["SMA_20"] = stock_data["Close"].rolling(window=20).mean()
stock_data["SMA_50"] = stock_data["Close"].rolling(window=50).mean()

# Plot stock closing prices
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data["Close"], label=f"{ticker_symbol} Closing Price", color='blue', linestyle='-', linewidth=2)
plt.plot(stock_data.index, stock_data["SMA_20"], label="20-Day SMA", color='orange', linestyle='--', linewidth=2)
plt.plot(stock_data.index, stock_data["SMA_50"], label="50-Day SMA", color='red', linestyle='-.', linewidth=2)

# Labels and title
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.title(f"{ticker_symbol} Stock Price Chart with SMA")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid for better visibility

# Show the graph
plt.show()
