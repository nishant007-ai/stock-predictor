import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from textblob import TextBlob
import requests

# 📄 Load Data
df = pd.read_csv("ipo_data.csv")

# 📊 Preprocess
df["Price_Mid"] = (df["Price_Low"] + df["Price_High"]) / 2
sector_map = {sector: idx for idx, sector in enumerate(df["Sector"].unique())}
df["Sector_Encoded"] = df["Sector"].map(sector_map)

# 🎯 Features
X = df[["Price_Mid", "GMP", "QIB", "NII", "Retail", "Total_Sub", "Sector_Encoded"]]
y = df["Listing_Gain"]

# 🎓 Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
score = r2_score(y_test, model.predict(X_test))

# 🌐 Streamlit UI
st.set_page_config(page_title="🚀 IPO Predictor", layout="centered")
st.title("📦 IPO Listing Gain Predictor (India 🇮🇳)")
st.info(f"🎯 Model R² Score: **{score:.2f}**")

# 📝 Input Section
st.subheader("🔢 Enter IPO Details")
ipo_name = st.text_input("IPO Name", "Tata Tech")
price_low = st.number_input("Price Band Low", 100)
price_high = st.number_input("Price Band High", 120)
gmp = st.number_input("GMP (Grey Market Premium)", 0)
qib = st.number_input("QIB Subscription (x)", 10.0)
nii = st.number_input("NII Subscription (x)", 5.0)
retail = st.number_input("Retail Subscription (x)", 3.0)
sector = st.selectbox("Sector", list(sector_map.keys()))

# 📊 Prepare Input
price_mid = (price_low + price_high) / 2
total_sub = qib + nii + retail
sector_encoded = sector_map.get(sector, 0)
input_data = [[price_mid, gmp, qib, nii, retail, total_sub, sector_encoded]]

# 🔮 Prediction
pred = model.predict(input_data)[0]
st.success(f"📈 Predicted Listing Gain for **{ipo_name}**: **{pred:.2f}%**")

# 🧠 News Sentiment (Optional)
st.subheader("📰 News Sentiment")
def fetch_sentiment(keyword):
    url = f"https://newsapi.org/v2/everything?q={keyword}&language=en&sortBy=publishedAt&pageSize=5&apiKey=3031443e554a4dd9b238c677207556ae"
    try:
        res = requests.get(url)
        articles = res.json().get("articles", [])
        sentiments = [TextBlob(a["title"]).sentiment.polarity for a in articles]
        return np.mean(sentiments) if sentiments else 0
    except:
        return 0

if st.button("🔍 Analyze Sentiment"):
    sentiment = fetch_sentiment(ipo_name)
    st.write(f"🧠 Sentiment Score: **{sentiment:.3f}**")
