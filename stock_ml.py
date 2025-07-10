import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import ta

# ✅ Add advanced technical indicators
def add_technical_indicators(df):
    df = df.copy()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['RSI'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    df['MACD'] = ta.trend.MACD(close=df['close']).macd()

    bb = ta.volatility.BollingerBands(close=df['close'])
    df['BB_MIDDLE'] = bb.bollinger_mavg()
    df['BB_UPPER'] = bb.bollinger_hband()
    df['BB_LOWER'] = bb.bollinger_lband()

    df['ATR'] = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['ROC'] = ta.momentum.ROCIndicator(close=df['close']).roc()
    df['Stochastic'] = ta.momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close']).stoch()
    df['Returns'] = df['close'].pct_change()
    df['Close_Lag_1'] = df['close'].shift(1)
    df['Close_Lag_2'] = df['close'].shift(2)
    df['Date_Ordinal'] = df.index.map(lambda d: d.toordinal())
    df['Prev_Close'] = df['close'].shift(1)

    return df.dropna()

# ✅ Prepare features and target
def prepare_features(df):
    features = [
        'Date_Ordinal', 'SMA_10', 'SMA_20', 'RSI', 'MACD',
        'BB_MIDDLE', 'BB_UPPER', 'BB_LOWER', 'ATR', 'ROC',
        'Stochastic', 'Returns', 'Close_Lag_1', 'Close_Lag_2',
        'Prev_Close', 'volume'
    ]
    X = df[features]
    y = df['close']
    return X, y

# ✅ Train model
def train_model(X, y, model_type='rf'):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=250, random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingRegressor(n_estimators=250, learning_rate=0.1, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, scaler, r2

# ✅ Predict future prices (with indicator updates)
def better_future_prediction(df, model, scaler, days=5):
    df_sim = df.copy()
    future_preds = []
    future_dates = []

    for i in range(days):
        df_sim = add_technical_indicators(df_sim)
        X, _ = prepare_features(df_sim)

        last_row = pd.DataFrame([X.iloc[-1]], columns=X.columns)
        last_row_scaled = scaler.transform(last_row)
        pred = model.predict(last_row_scaled)[0]

        # Store prediction
        next_date = df_sim.index[-1] + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:  # skip weekends
            next_date += pd.Timedelta(days=1)

        future_preds.append(pred)
        future_dates.append(next_date.strftime("%Y-%m-%d"))

        # Simulate next day row
        new_row = df_sim.iloc[-1].copy()
        new_row.name = next_date
        new_row['close'] = pred
        new_row['Prev_Close'] = df_sim['close'].iloc[-1]
        new_row['volume'] *= np.random.uniform(0.98, 1.02)

        df_sim = pd.concat([df_sim, pd.DataFrame([new_row])])

    return pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds})
