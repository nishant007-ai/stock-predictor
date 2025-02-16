import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 🎯 Streamlit UI Title
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction App")

# 📌 Sidebar for Stock Selection
st.sidebar.header("Stock Analysis Options")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, GOOGL):", "AAPL").upper()

# Initialize session state for stock data
if "stock_data" not in st.session_state:
    st.session_state.stock_data = yf.download(ticker_symbol, start="2024-01-01", end="2025-02-16").dropna()

# 📌 Refresh Button for Live Updates
if st.button("🔄 Refresh Data"):
    st.session_state.stock_data = yf.download(ticker_symbol, period="1d", interval="1m").dropna()
    st.experimental_rerun()  # ✅ Ensures the app refreshes properly

# Use session state data
stock_data = st.session_state.stock_data

# 🔎 Debugging: Display available columns
st.write("Available Columns:", stock_data.columns)

# 🗒 Show Data
st.subheader(f"📊 {ticker_symbol} Stock Data (Last 5 Days)")
st.write(stock_data.tail())

# 📉 Closing Price Chart with Matplotlib
st.subheader(f"📉 {ticker_symbol} Closing Price Chart")
plt.figure(figsize=(10, 5))
plt.plot(stock_data.index, stock_data["Close"], label="Closing Price", color='blue')
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
st.pyplot(plt)

# 📊 Closing Price Chart with Plotly
fig = px.line(stock_data, x=stock_data.index, y="Close", title=f"{ticker_symbol} Closing Price Over Time")
st.plotly_chart(fig)

# 🧠 Prepare Data for Prediction
stock_data['Date'] = stock_data.index
stock_data['Date'] = stock_data['Date'].map(pd.Timestamp.toordinal)  # Convert Date to Numeric
X = stock_data[['Date']]
y = stock_data['Close']

# 📚 Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 🔮 Predict Next 5 Days
future_dates = np.arange(X_test.max() + 1, X_test.max() + 6).reshape(-1, 1)
predictions = model.predict(future_dates)

# 📈 Display Predictions
st.subheader("📈 Predicted Stock Prices for the Next 5 Days")
for i, price in enumerate(predictions, 1):
    st.write(f"📅 Day {i}: **${price:.2f}**")
