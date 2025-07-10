if "last_ticker" not in st.session_state or st.session_state.last_ticker != ticker_symbol:
    try:
        data = yf.download(ticker_symbol, start="2024-01-01", end="2025-02-16", progress=False)
        if data.empty:
            data = yf.download(ticker_symbol, period="7d", interval="1d", progress=False)
        if data.empty:
            st.error("❌ No stock data found.")
            st.stop()
        st.session_state.stock_data = data.dropna()
        st.session_state.last_ticker = ticker_symbol
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        st.stop()
else:
    data = st.session_state.stock_data
