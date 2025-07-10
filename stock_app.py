import streamlit as st
from twelvedata import TDClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import timedelta
import ta
import requests
import yfinance as yf
from textblob import TextBlob
from nsetools import Nse  

# 📡 API Keys
API_KEY = "aad4bc8137e84c7684520458c952d10a"
td = TDClient(apikey=API_KEY)
NEWS_API_KEY = "3031443e554a4dd9b238c677207556ae"

# ⚙️ Initialize APIs
td = TDClient(apikey=API_KEY)
nse = Nse()  

# 🎯 Page Setup
st.set_page_config(page_title="📈 Stock Prediction", layout="wide")
st.title("📊 Advanced Stock Price Prediction App")

# 📌 Sidebar Ticker Input
st.sidebar.header("📈 Choose Stock")

# ✅ Get list of all NSE stock symbols
try:
    stock_dict = nse.get_stock_codes()
    stock_dict.pop('SYMBOL', None)
    all_stocks = sorted(stock_dict.keys())
except:
    all_stocks = [
        # 🌍 Global Stocks (Twelve Data Supported)
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
        "INTC", "AMD", "BA", "JPM", "DIS", "PEP", "NKE", "ORCL", "PYPL",
        "KO", "MRNA", "ADBE",  # ← fixed comma here!

        # 🇮🇳 Indian Stocks (YFinance Supported with .NS)
        "TCS", "INFY", "WIPRO", "HDFCBANK", "ICICIBANK", "AXISBANK",
        "KOTAKBANK", "SBIN", "HINDUNILVR", "ITC", "ONGC", "NTPC", "POWERGRID",
        "BHARTIARTL", "BAJFINANCE", "BAJAJFINSV", "ADANIENT", "ADANIPORTS", 
        "HCLTECH", "LT", "TATAMOTORS", "TATASTEEL", "COALINDIA", "SUNPHARMA", 
        "TECHM", "VEDL", "UPL", "BPCL", "IOC", "HEROMOTOCO", "EICHERMOT", 
        "ULTRACEMCO", "GRASIM", "JSWSTEEL", "DIVISLAB", "ASIANPAINT", 
        "CIPLA", "BRITANNIA", "SHREECEM", "MARUTI", "SUZLON"
    ]


selected_symbol = st.sidebar.selectbox("Select Indian Stock:", all_stocks)
custom_input = st.sidebar.text_input("Or enter custom ticker (e.g., AAPL, TSLA):", selected_symbol)
ticker = custom_input.strip().upper()

# 📥 Fetch Stock Data
@st.cache_data(show_spinner=False)
def fetch_data(symbol):
    try:
        df = td.time_series(symbol=symbol, interval="1day", outputsize=200).as_pandas()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        return pd.DataFrame()

df = fetch_data(ticker)
if df.empty:
    st.stop()

# ✅ Latest Price
latest_price = df["close"].iloc[-1]
st.success(f"✅ Latest Closing Price for **{ticker}**: ${latest_price:.2f}")

# 📰 News Sentiment + Headlines
def fetch_news_sentiment_and_headlines(stock_name):
    url = f"https://newsapi.org/v2/everything?q={stock_name}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    try:
        res = requests.get(url)
        articles = res.json().get("articles", [])
        data = []

        for article in articles:
            title = article["title"]
            url = article["url"]
            published = article["publishedAt"]
            date_time = pd.to_datetime(published).strftime("%Y-%m-%d %H:%M")

            sentiment = TextBlob(title).sentiment.polarity

            if sentiment > 0.1:
                emoji = "🟢"
            elif sentiment < -0.1:
                emoji = "🔴"
            else:
                emoji = "⚪"

            data.append({
                "emoji": emoji,
                "sentiment": sentiment,
                "title": title,
                "url": url,
                "datetime": date_time
            })

        return data
    except:
        return []

# 📦 Show Sentiment Results
with st.expander("📰 News Sentiment & Headlines"):
    news_data = fetch_news_sentiment_and_headlines(ticker)

    if news_data:
        sentiments = [item["sentiment"] for item in news_data]
        st.write(f"🧠 Avg. Sentiment Polarity: {np.mean(sentiments):.3f}")
        st.bar_chart(sentiments)

        st.markdown("### 🗞 Latest Headlines with Sentiment:")
        for item in news_data:
            st.markdown(
                f"{item['emoji']} **[{item['title']}]({item['url']})**  \n"
                f"📅 {item['datetime']}  \n"
                f"📊 Sentiment Score: `{item['sentiment']:.2f}`"
            )
    else:
        st.warning("⚠️ Could not fetch sentiment data or headlines.")


# 📊 Add Technical Indicators
df["SMA_10"] = ta.trend.sma_indicator(df["close"], window=10)
df["SMA_20"] = ta.trend.sma_indicator(df["close"], window=20)
df["RSI"] = ta.momentum.rsi(df["close"], window=14)
df["MACD"] = ta.trend.macd(df["close"])
df["BB_High"] = ta.volatility.BollingerBands(df["close"]).bollinger_hband()
df["BB_Low"] = ta.volatility.BollingerBands(df["close"]).bollinger_lband()
df["ATR"] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"]).average_true_range()
df["Prev_Close"] = df["close"].shift(1)
df["Date_Ordinal"] = df.index.map(pd.Timestamp.toordinal)
df.dropna(inplace=True)

# 📋 Show Recent Data
st.subheader(f"📊 Recent Stock Data for {ticker}")
st.dataframe(df.tail())

# 📈 Closing Price Plot
fig = px.line(df, x=df.index, y="close", title=f"{ticker} Closing Price Over Time")
st.plotly_chart(fig)

# 📈 Moving Average Plot
fig_ma = px.line(df, x=df.index, title="📈 Price with SMA & Bollinger Bands")
fig_ma.add_scatter(x=df.index, y=df['close'], mode='lines', name='Close')
fig_ma.add_scatter(x=df.index, y=df['SMA_10'], mode='lines', name='SMA 10')
fig_ma.add_scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20')
fig_ma.add_scatter(x=df.index, y=df['BB_High'], mode='lines', name='Bollinger High')
fig_ma.add_scatter(x=df.index, y=df['BB_Low'], mode='lines', name='Bollinger Low')
st.plotly_chart(fig_ma)

# 🎯 Prepare Features
features = [
    "Date_Ordinal", "SMA_10", "SMA_20", "RSI", "MACD",
    "BB_High", "BB_Low", "ATR", "Prev_Close", "volume"
]
X = df[features].copy()
y = df["close"]

# 🎓 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Train Random Forest
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
score = r2_score(y_test, model.predict(X_test))
st.info(f"📊 Model R² Score: **{score:.4f}**")

# 🔮 Predict Future Dates (Weekdays only)
today = pd.to_datetime("today").normalize()
future_dates = []
d = 1
while len(future_dates) < 5:
    next_day = today + timedelta(days=d)
    if next_day.weekday() < 5:
        future_dates.append(next_day)
    d += 1

# 🧠 Predict Using Last Known Row
last_row = X.iloc[-1].copy()
predictions = []

for f_date in future_dates:
    next_features = last_row.copy()
    next_features["Date_Ordinal"] = f_date.toordinal()

    for key in ["RSI", "MACD", "ATR"]:
        next_features[key] *= (1 + np.random.uniform(-0.01, 0.01))

    for key in ["SMA_10", "SMA_20", "BB_High", "BB_Low"]:
        next_features[key] *= (1 + np.random.uniform(-0.002, 0.002))

    next_features["Prev_Close"] = predictions[-1] if predictions else latest_price

    pred = model.predict([next_features])[0]
    predictions.append(pred)
# 🧠 Predict Using Last Known Row — Realistic High-Margin Simulation
last_row = X.iloc[-1].copy()
predictions = []

# Sentiment boost
news_sentiments = fetch_news_sentiment_and_headlines(ticker)
avg_sentiment = np.mean([item["sentiment"] for item in news_sentiments]) if news_sentiments else 0
sentiment_boost = avg_sentiment * 5  # You can adjust this

# Set prediction swing range (force at least ±2% daily)
price_volatility_factor = 0.02  # 2% minimum swing per day

for f_date in future_dates:
    next_features = last_row.copy()
    next_features["Date_Ordinal"] = f_date.toordinal()

    # Simulate indicator movements more aggressively
    next_features["RSI"] *= (1 + np.random.uniform(-0.10, 0.10))
    next_features["MACD"] *= (1 + np.random.uniform(-0.10, 0.10))
    next_features["ATR"] *= (1 + np.random.uniform(-0.20, 0.20))
    next_features["SMA_10"] *= (1 + np.random.uniform(-0.02, 0.02))
    next_features["SMA_20"] *= (1 + np.random.uniform(-0.02, 0.02))
    next_features["BB_High"] *= (1 + np.random.uniform(-0.02, 0.02))
    next_features["BB_Low"] *= (1 + np.random.uniform(-0.02, 0.02))
    next_features["volume"] *= (1 + np.random.uniform(-0.25, 0.25))

    # Set previous close
    next_features["Prev_Close"] = predictions[-1] if predictions else latest_price

    pred = model.predict([next_features])[0]

    # Force swing at least 2%
    last_price = predictions[-1] if predictions else latest_price
    min_swing = last_price * price_volatility_factor
    if abs(pred - last_price) < min_swing:
        direction = 1 if np.random.rand() > 0.5 else -1
        pred = last_price + direction * min_swing

    # Apply sentiment impact
    pred += sentiment_boost

    predictions.append(pred)

    # Update last row for next prediction
    last_row = next_features.copy()
    last_row["Prev_Close"] = pred


    

# 📈 Display Trend Info
if predictions[0] < latest_price:
    st.warning("📉 First predicted day is lower than current price. Possible dip.")
else:
    st.success("📈 First predicted day is higher than current price. Possible uptrend.")

# 🔮 Prediction Display
st.subheader("🔮 Predicted Prices (Next 5 Days)")
for i, price in enumerate(predictions):
    st.write(f"📆 Day {i + 1}: **${price:.2f}**")

st.subheader("📅 Prediction by Date")
for d, price in zip(future_dates, predictions):
    st.write(f"📅 {d.date()}: **${price:.2f}**")

# 📊 Chart: Actual vs Predicted
future_df = pd.DataFrame({"Date": future_dates, "Predicted": predictions}).set_index("Date")
recent_df = df[["close"]].iloc[-5:]
recent_df.columns = ["Predicted"]
combined = pd.concat([recent_df, future_df])
st.subheader("📊 Actual (Last 5) vs Predicted (Next 5)")
st.line_chart(combined)

# 🚀 IPO Prediction Section
st.markdown("---")
st.header("🚀 Predict Upcoming IPO Performance")

# 📦 Indian IPO Pre-Fill Options (Expanded)
indian_ipos = {
    "OYO Rooms": {"price": 85.0, "volume": 22.5},
    "Mobikwik": {"price": 100.0, "volume": 15.0},
    "Pharmeasy": {"price": 120.0, "volume": 12.0},
    "Ola Electric": {"price": 105.0, "volume": 18.0},
    "Tata Technologies": {"price": 500.0, "volume": 45.0},
    "LIC": {"price": 949.0, "volume": 221.0},
    "Zomato": {"price": 76.0, "volume": 120.0},
    "Nykaa": {"price": 1125.0, "volume": 18.5},
    "Delhivery": {"price": 487.0, "volume": 45.0},
    "Paytm": {"price": 2150.0, "volume": 85.0},
    "Suzlon Energy": {"price": 100.0, "volume": 100.0},
    "Adani Wilmar": {"price": 230.0, "volume": 45.0},
    "IRCTC": {"price": 320.0, "volume": 20.0},
    "Tata Motors (Potential)": {"price": 620.0, "volume": 80.0}
}

st.markdown("## 🇮🇳 Upcoming Indian IPOs")
selected_ipo = st.selectbox("📌 Select an Indian IPO to autofill details", ["Custom Input"] + list(indian_ipos.keys()))

# Set default values based on selection
if selected_ipo != "Custom Input":
    default_name = selected_ipo
    default_price = indian_ipos[selected_ipo]["price"]
    default_volume = indian_ipos[selected_ipo]["volume"]
else:
    default_name = "IPOX"
    default_price = 100.0
    default_volume = 10.0

with st.form("ipo_form"):
    st.subheader("📥 Enter IPO Details")

    ipo_name = st.text_input("IPO Stock Name or Symbol", default_name)
    ipo_price = st.number_input("Expected Listing Price ($)", value=default_price)
    ipo_volume = st.number_input("Expected Volume (in millions)", value=default_volume)
    sentiment_score = st.slider("Market Sentiment Score", -1.0, 1.0, 0.1)

    submitted = st.form_submit_button("🔮 Predict IPO Performance")

if submitted:
    # Simple heuristic: first-day price movement
    impact = (sentiment_score * 10) + (ipo_volume * 0.5)
    predicted_close = ipo_price + impact

    st.success(f"🎯 **Predicted 1st Day Closing Price for {ipo_name}: ${predicted_close:.2f}**")

    if predicted_close > ipo_price:
        st.info("📈 Prediction suggests the IPO may perform positively on launch.")
    elif predicted_close < ipo_price:
        st.warning("📉 Prediction suggests the IPO might dip below listing price.")
    else:
        st.info("⚖️ Prediction suggests neutral performance on launch.")

    st.markdown("### 🧠 Calculation Breakdown")
    st.markdown(f"- 🏷 Listing Price: ${ipo_price}")
    st.markdown(f"- 📦 Volume Factor: {ipo_volume * 0.5}")
    st.markdown(f"- 💬 Sentiment Factor: {sentiment_score * 10}")
