# ₿ BTC/USDT Next-Hour Forecaster
### AlphaI × Polaris Build Challenge

A live Bitcoin price range forecaster built with Geometric Brownian Motion (GBM) simulation. Instead of predicting an exact price, it predicts a **95% confidence interval** for where BTC will land one hour from now.

---

## Live Dashboard

🔗 **https://btc-forecaster-aboro.streamlit.app/**

The dashboard shows:
- Current BTC price (live from Binance)
- Predicted 95% range for the next hour
- Last 50 bars with predicted range as a shaded ribbon
- Backtest metrics (coverage, avg width, Winkler score) as headline numbers

---

## Results

| Metric | Value |
|---|---|
| Bars tested | 720 (30 days of hourly data) |
| Coverage (target ~0.95) | **0.9542** |
| Avg range width | $1,386.62 |
| Avg Winkler score | $1,767.67 |

---

## How It Works

### Data
Hourly BTCUSDT candles fetched from Binance's public API — no API key required.
```
https://data-api.binance.vision/api/v3/klines
```

### Model — GBM with Student-t
At each bar, the model:
1. Computes log returns from price history
2. Estimates recent volatility from the **last 24 bars** (volatility clustering)
3. Fits a **Student-t distribution** to capture fat tails
4. Simulates 10,000 possible next prices via Monte Carlo
5. Reads off the 2.5th and 97.5th percentiles as the 95% range

### Three Key Concepts

**No peeking** — when predicting bar N, only data up to bar N−1 is used. No future data leaks into any prediction.

**Volatility clustering** — recent volatility (last 24 hours) drives range width. Calm periods → narrow range. Volatile periods → wider range.

**Fat tails** — Student-t distribution instead of normal. BTC has more frequent extreme moves than a normal bell curve would predict, so this prevents systematic under-coverage.

---

## Project Structure

```
btc-forecaster/
├── app.py                   # Streamlit dashboard
├── requirements.txt         # Python dependencies
├── backtest.ipynb           # Full backtest notebook (Colab)
└── backtest_results.jsonl   # 720 predictions with actuals
```

---

## Backtest Logic

```python
for each bar i in last 720 bars:
    history = prices[:i]              # only past data — no peeking
    low, high = gbm_predict(history)  # predict 95% range
    actual    = prices[i+1]           # reveal actual next bar
    record(low, high, actual, winkler_score)
```

### Winkler Score
A single number combining accuracy and tightness:
- If actual falls **inside** the range → score = range width
- If actual falls **outside** the range → score = range width + (2/α) × miss distance

Lower Winkler = better forecaster.

---

## Running Locally

```bash
pip install streamlit requests numpy pandas scipy plotly
streamlit run app.py
```

---

## Dependencies

```
streamlit
requests
numpy
pandas
scipy
plotly
```

---

## Deployment

Hosted on [Streamlit Community Cloud](https://streamlit.io/cloud) — free tier, stays live for 7+ days.