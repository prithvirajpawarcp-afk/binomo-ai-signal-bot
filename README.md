# AI Multi-Indicator Signal Bot V1 — REAL OHLC MODE

## What changed
This version adds a live-data endpoint using Twelve Data's `/time_series` API:
- API-key input in the mobile dashboard
- EUR/USD by default
- 1-minute candles by default
- 60+ completed candles required
- newest/in-progress bar is excluded before analysis
- 11-indicator weighted engine
- signal history
- no demo/random candle generation in Live Mode

Twelve Data documents `/time_series`, `apikey`, `symbol`, `interval`, and `outputsize`; supported intervals include 1min, 5min, 15min, 30min, 45min, 1h, etc.

## Run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Termux/Android
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000`.

## Live endpoint

`POST /api/live-signal`

Example:

```json
{
  "api_key": "YOUR_TWELVE_DATA_KEY",
  "symbol": "EUR/USD",
  "interval": "1min",
  "outputsize": 120
}
```

The API key is not saved in signal history and is not returned in the response.

## Important
- This is genuine provider OHLC data when the request succeeds; it is NOT a random/synthetic signal.
- It does NOT access Binomo's private or proprietary OTC feed.
- Twelve Data EUR/USD and a broker's OTC quote can differ. Therefore the dashboard does not call the result a guaranteed "Binomo OTC signal."
- A real feed still cannot guarantee a winning trade.
- The signal's confidence is an indicator-agreement score, not a probability of profit.
- Use paper testing/backtesting before risking money.
