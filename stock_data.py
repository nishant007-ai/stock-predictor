import yfinance as yf

# Enter the stock ticker symbol
ticker_symbol = input("Enter stock ticker symbol: ").strip().upper()

# Download the stock data
stock_data = yf.download(ticker_symbol, start="2024-01-01", end="2025-02-16")

# Save to CSV file
csv_filename = f"{ticker_symbol}_stock_data.csv"
stock_data.to_csv(csv_filename)

print(f"✅ Data saved to {csv_filename}")

stock_data["50-day MA"] = stock_data["Close"].rolling(window=50).mean()
print(stock_data.tail())  # Show latest values with the moving average
