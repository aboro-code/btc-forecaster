import streamlit as st
import requests
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go

# ── page config ────────────────────────────────────────────
st.set_page_config(page_title="BTC Forecaster", layout="wide")
st.title("₿ BTC/USDT — Next Hour Forecast")

# ── data fetch ─────────────────────────────────────────────
@st.cache_data(ttl=300)  # refresh every 5 minutes
def get_btc_hourly(n_bars=500):
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": n_bars}
    r = requests.get(url, params=params)
    df = pd.DataFrame(r.json(), columns=[
        "open_time","open","high","low","close",
        "volume","close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    return df["close"]

# ── GBM predictor (same as backtest) ───────────────────────
def predict_next_hour(price_series, n_sims=10_000):
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    recent_vol = log_ret.iloc[-24:].std() * 1.3   # same scaling as backtest
    mu = log_ret.iloc[-24:].mean()
    nu, _, _ = stats.t.fit(log_ret.iloc[-100:])
    nu = max(4, nu)
    S0 = price_series.iloc[-1]
    Z = np.random.standard_t(nu, size=n_sims)
    Z = Z * np.sqrt((nu - 2) / nu)
    next_prices = S0 * np.exp((mu - 0.5 * recent_vol**2) + recent_vol * Z)
    low  = np.percentile(next_prices, 2.5)
    high = np.percentile(next_prices, 97.5)
    return low, high, S0

# ── load data + predict ────────────────────────────────────
with st.spinner("Fetching latest BTC data..."):
    prices = get_btc_hourly(500)

low, high, current = predict_next_hour(prices)

# ── headline metrics ───────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current BTC Price", f"${current:,.2f}")
col2.metric("Predicted Low",     f"${low:,.2f}")
col3.metric("Predicted High",    f"${high:,.2f}")
col4.metric("Coverage (backtest)", "95.42%")
col5.metric("Avg Winkler Score",   "$1,767.67")

st.divider()

# ── chart: last 50 bars + predicted range ribbon ───────────
last_50 = prices.iloc[-50:]
next_time = last_50.index[-1] + pd.Timedelta(hours=1)

fig = go.Figure()

# price line
fig.add_trace(go.Scatter(
    x=last_50.index, y=last_50.values,
    mode="lines", name="BTC Price",
    line=dict(color="#F7931A", width=2)
))

# shaded ribbon for predicted range
fig.add_trace(go.Scatter(
    x=[last_50.index[-1], next_time, next_time, last_50.index[-1]],
    y=[current, high, low, current],
    fill="toself",
    fillcolor="rgba(99, 110, 250, 0.2)",
    line=dict(color="rgba(99,110,250,0)"),
    name="95% Predicted Range"
))

# high/low lines
fig.add_trace(go.Scatter(
    x=[last_50.index[-1], next_time], y=[current, high],
    mode="lines", line=dict(color="green", dash="dash"),
    name=f"High ${high:,.0f}"
))
fig.add_trace(go.Scatter(
    x=[last_50.index[-1], next_time], y=[current, low],
    mode="lines", line=dict(color="red", dash="dash"),
    name=f"Low ${low:,.0f}"
))

fig.update_layout(
    title="Last 50 Bars + Next Hour Prediction",
    xaxis_title="Time (UTC)",
    yaxis_title="Price (USDT)",
    hovermode="x unified",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"Data refreshes every 5 minutes. Last fetch: {prices.index[-1]} UTC")