from __future__ import annotations

from pathlib import Path
from typing import List, Literal
from datetime import datetime, timezone
import json
import math
import statistics
import asyncio
import urllib.parse
import urllib.request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
HISTORY_FILE = BASE / "signal_history.json"

app = FastAPI(title="AI Multi-Indicator Signal Bot V1", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. Restrict this in production.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Candle(BaseModel):
    time: str | int | float
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(default=0, ge=0)


class CandleRequest(BaseModel):
    symbol: str = "EUR/USD"
    timeframe: str = "1m"
    candles: List[Candle] = Field(min_length=60)


class SignalRequest(CandleRequest):
    source: str = "external_ohlc"


class LiveSignalRequest(BaseModel):
    api_key: str = Field(min_length=8)
    symbol: str = "EUR/USD"
    interval: Literal["1min", "5min", "15min", "30min", "1h"] = "1min"
    outputsize: int = Field(default=120, ge=60, le=5000)
    source: str = "twelve_data"




WEIGHTS = {
    "RSI": 10,
    "Stochastic": 10,
    "CCI": 8,
    "Bollinger": 8,
    "Alligator": 10,
    "Fractal": 8,
    "MACD": 12,
    "Price Action": 12,
    "Liquidity": 10,
    "Parabolic SAR": 6,
    "ATR": 6,
}


def closes(cs): return [x.close for x in cs]
def highs(cs): return [x.high for x in cs]
def lows(cs): return [x.low for x in cs]


def sma(values, n):
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def ema_series(values, n):
    if len(values) < n:
        return []
    k = 2 / (n + 1)
    out = [sum(values[:n]) / n]
    for v in values[n:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def ema(values, n):
    s = ema_series(values, n)
    return s[-1] if s else None


def stddev(values, n):
    if len(values) < n:
        return None
    return statistics.pstdev(values[-n:])


def rsi(values, n=14):
    if len(values) < n + 1:
        return None
    gains, losses = [], []
    for i in range(-n, 0):
        d = values[i] - values[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag, al = sum(gains) / n, sum(losses) / n
    if al == 0:
        return 100.0
    rs = ag / al
    return 100 - 100 / (1 + rs)


def stochastic(cs, n=14, smooth=3):
    if len(cs) < n + smooth:
        return None, None
    ks = []
    for end in range(n, len(cs) + 1):
        window = cs[end - n:end]
        hh, ll = max(highs(window)), min(lows(window))
        k = 50 if hh == ll else 100 * (window[-1].close - ll) / (hh - ll)
        ks.append(k)
    k = sum(ks[-smooth:]) / smooth
    d = sum(ks[-smooth*2:-smooth] if len(ks) >= smooth*2 else ks[:smooth]) / min(smooth, len(ks))
    return k, d


def cci(cs, n=20):
    if len(cs) < n:
        return None
    tp = [(x.high + x.low + x.close) / 3 for x in cs]
    mean = sum(tp[-n:]) / n
    dev = sum(abs(x - mean) for x in tp[-n:]) / n
    return 0 if dev == 0 else (tp[-1] - mean) / (0.015 * dev)


def bollinger(cs, n=20, mult=2):
    vals = closes(cs)
    if len(vals) < n:
        return None
    mid = sma(vals, n)
    sd = stddev(vals, n)
    return mid, mid + mult * sd, mid - mult * sd


def alligator(cs):
    # V1 approximation: SMA-smoothed Alligator lines.
    # Bill Williams uses smoothed averages with forward shifts; this version
    # compares current unshifted proxy lines to avoid future-looking data.
    vals = [(x.high + x.low) / 2 for x in cs]
    jaw = sma(vals, 13)
    teeth = sma(vals, 8)
    lips = sma(vals, 5)
    return jaw, teeth, lips


def macd(cs, fast=12, slow=26, signal=9):
    vals = closes(cs)
    mf, ms = ema_series(vals, fast), ema_series(vals, slow)
    if not mf or not ms:
        return None, None, None
    # Align fast EMA to slow EMA timeline.
    offset = slow - fast
    fast_aligned = mf[offset:]
    n = min(len(fast_aligned), len(ms))
    line = [fast_aligned[-n+i] - ms[-n+i] for i in range(n)]
    sig = ema_series(line, signal)
    if not sig:
        return None, None, None
    return line[-1], sig[-1], line[-1] - sig[-1]


def fractal_signal(cs, lookback=2):
    if len(cs) < lookback * 2 + 3:
        return 0, None, None
    # Most recently CONFIRMED fractals, excluding the current candle.
    bull = bear = None
    last = len(cs) - 1 - lookback
    for i in range(last, lookback - 1, -1):
        h = cs[i].high
        l = cs[i].low
        if all(h > cs[j].high for j in range(i-lookback, i+lookback+1) if j != i):
            bear = h
            break
    for i in range(last, lookback - 1, -1):
        l = cs[i].low
        if all(l < cs[j].low for j in range(i-lookback, i+lookback+1) if j != i):
            bull = l
            break
    c = cs[-1].close
    if bear is not None and c > bear:
        return 1, bull, bear
    if bull is not None and c < bull:
        return -1, bull, bear
    return 0, bull, bear


def price_action(cs):
    if len(cs) < 4:
        return 0, "insufficient"
    a, b, c = cs[-3], cs[-2], cs[-1]
    body = abs(c.close - c.open)
    rng = max(c.high - c.low, 1e-12)
    upper = c.high - max(c.open, c.close)
    lower = min(c.open, c.close) - c.low
    bull_engulf = b.close < b.open and c.close > c.open and c.close >= b.open and c.open <= b.close
    bear_engulf = b.close > b.open and c.close < c.open and c.open >= b.close and c.close <= b.open
    bull_pin = lower > body * 2 and c.close > c.open
    bear_pin = upper > body * 2 and c.close < c.open
    higher = c.high > b.high and c.low > b.low
    lower_seq = c.high < b.high and c.low < b.low
    if bull_engulf or bull_pin or higher:
        return 1, "bullish"
    if bear_engulf or bear_pin or lower_seq:
        return -1, "bearish"
    return 0, "neutral"


def atr(cs, n=14):
    if len(cs) < n + 1:
        return None
    trs = []
    for i in range(1, len(cs)):
        x, p = cs[i], cs[i-1]
        trs.append(max(x.high-x.low, abs(x.high-p.close), abs(x.low-p.close)))
    return sum(trs[-n:]) / n


def psar(cs, step=0.02, max_af=0.2):
    # Standard-style PSAR implementation, returning current SAR and trend.
    if len(cs) < 5:
        return None, 0
    bull = True
    sar = cs[0].low
    ep = cs[0].high
    af = step
    for i in range(1, len(cs)):
        prev = cs[i-1]
        cur = cs[i]
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, prev.low)
            if i >= 2:
                sar = min(sar, cs[i-2].low)
            if cur.low < sar:
                bull = False
                sar = ep
                ep = cur.low
                af = step
            elif cur.high > ep:
                ep = cur.high
                af = min(max_af, af + step)
        else:
            sar = max(sar, prev.high)
            if i >= 2:
                sar = max(sar, cs[i-2].high)
            if cur.high > sar:
                bull = True
                sar = ep
                ep = cur.high
                af = step
            elif cur.low < ep:
                ep = cur.low
                af = min(max_af, af + step)
    return sar, (1 if bull else -1)


def liquidity_signal(cs, n=20):
    if len(cs) < n + 3:
        return 0, "insufficient"
    # Heuristic liquidity sweep: current candle pierces prior range then closes back inside.
    prior = cs[-n-1:-1]
    ph, pl = max(x.high for x in prior), min(x.low for x in prior)
    c = cs[-1]
    if c.low < pl and c.close > pl:
        return 1, "sell-side sweep + reclaim"
    if c.high > ph and c.close < ph:
        return -1, "buy-side sweep + rejection"
    return 0, "no sweep"


def score_indicator(name, direction, value, detail):
    w = WEIGHTS[name]
    return {"name": name, "weight": w, "direction": direction, "value": value, "detail": detail}


def analyze(cs):
    if len(cs) < 60:
        raise ValueError("At least 60 OHLC candles are required.")

    c = cs[-1].close
    items = []

    rv = rsi(closes(cs))
    rdir = 1 if rv is not None and rv > 50 else -1 if rv is not None and rv < 50 else 0
    items.append(score_indicator("RSI", rdir, round(rv, 2) if rv is not None else None, "14-period; >50 bullish, <50 bearish"))

    k, d = stochastic(cs)
    sdir = 1 if k is not None and d is not None and k > d else -1 if k is not None and d is not None and k < d else 0
    items.append(score_indicator("Stochastic", sdir, {"k": round(k,2) if k is not None else None, "d": round(d,2) if d is not None else None}, "14/3"))

    cv = cci(cs)
    cdir = 1 if cv is not None and cv > 0 else -1 if cv is not None and cv < 0 else 0
    items.append(score_indicator("CCI", cdir, round(cv,2) if cv is not None else None, "20-period"))

    bb = bollinger(cs)
    bdir = 0
    if bb:
        mid, upper, lower = bb
        if c > mid: bdir = 1
        elif c < mid: bdir = -1
    items.append(score_indicator("Bollinger", bdir, {"mid": round(bb[0],6), "upper": round(bb[1],6), "lower": round(bb[2],6)} if bb else None, "20-period, 2σ"))

    jaw, teeth, lips = alligator(cs)
    adir = 1 if jaw is not None and lips > teeth > jaw else -1 if jaw is not None and lips < teeth < jaw else 0
    items.append(score_indicator("Alligator", adir, {"jaw": jaw, "teeth": teeth, "lips": lips}, "13/8/5 proxy"))

    fdir, bull_fr, bear_fr = fractal_signal(cs)
    items.append(score_indicator("Fractal", fdir, {"bull_fractal": bull_fr, "bear_fractal": bear_fr}, "2-left/2-right confirmed fractal"))

    ml, sl, hist = macd(cs)
    mdir = 1 if ml is not None and ml > sl else -1 if ml is not None and ml < sl else 0
    items.append(score_indicator("MACD", mdir, {"macd": ml, "signal": sl, "hist": hist}, "12/26/9"))

    pdir, pdetail = price_action(cs)
    items.append(score_indicator("Price Action", pdir, pdetail, "engulfing/pin/structure"))

    ldir, ldetail = liquidity_signal(cs)
    items.append(score_indicator("Liquidity", ldir, ldetail, "range sweep heuristic"))

    sar, psdir = psar(cs)
    items.append(score_indicator("Parabolic SAR", psdir, sar, "0.02 / 0.20"))

    av = atr(cs)
    # ATR is a volatility filter, not inherently bullish/bearish.
    # Give its points to the prevailing directional side only if ATR is non-zero.
    atr_dir = 0
    if av and av > 0:
        # Direction follows price action for scoring; if price action is neutral, no points.
        atr_dir = pdir
    items.append(score_indicator("ATR", atr_dir, av, "14-period volatility confirmation"))

    buy = sum(x["weight"] for x in items if x["direction"] == 1)
    sell = sum(x["weight"] for x in items if x["direction"] == -1)
    total = sum(WEIGHTS.values())

    # Only call a directional signal when one side reaches 65 and leads by at least 15 points.
    if buy >= 65 and buy - sell >= 15:
        signal = "BUY"
    elif sell >= 65 and sell - buy >= 15:
        signal = "SELL"
    else:
        signal = "NO TRADE"

    agreement = sum(1 for x in items if (x["direction"] == 1 if signal == "BUY" else x["direction"] == -1 if signal == "SELL" else x["direction"] != 0))
    dominant = max(buy, sell)
    confidence = round(100 * dominant / total)

    return {
        "signal": signal,
        "buy_score": buy,
        "sell_score": sell,
        "total_score": total,
        "confidence": confidence,
        "agreement_count": agreement,
        "indicator_count": len(items),
        "price": c,
        "atr": av,
        "indicators": items,
    }


def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return []


def save_history(record):
    history = load_history()
    history.insert(0, record)
    HISTORY_FILE.write_text(json.dumps(history[:100], indent=2))


def fetch_twelve_data(api_key: str, symbol: str, interval: str, outputsize: int):
    params = urllib.parse.urlencode({
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
    })
    url = "https://api.twelvedata.com/time_series?" + params
    req = urllib.request.Request(url, headers={"User-Agent": "AI-Signal-Bot-V1/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Market-data request failed: {e}")

    if payload.get("status") == "error" or "values" not in payload:
        msg = payload.get("message", "Twelve Data returned no candle data.")
        raise HTTPException(status_code=502, detail=msg)

    values = payload["values"]
    parsed = []
    for v in values:
        try:
            parsed.append(Candle(
                time=v["datetime"],
                open=float(v["open"]),
                high=float(v["high"]),
                low=float(v["low"]),
                close=float(v["close"]),
                volume=float(v.get("volume") or 0),
            ))
        except Exception:
            continue

    # Twelve Data returns bars with timestamps identifying the bar's opening time.
    # Sort oldest -> newest and remove the newest bar so the engine never analyzes
    # an in-progress candle.
    parsed.sort(key=lambda x: str(x.time))
    if len(parsed) > 60:
        parsed = parsed[:-1]

    if len(parsed) < 60:
        raise HTTPException(status_code=502, detail="Fewer than 60 completed candles were available.")

    return parsed, payload.get("meta", {})


@app.post("/api/live-signal")
def live_signal(req: LiveSignalRequest):
    candles, meta = fetch_twelve_data(req.api_key, req.symbol, req.interval, req.outputsize)
    result = analyze(candles)

    record = {
        "id": datetime.now(timezone.utc).isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": req.symbol,
        "timeframe": req.interval,
        "source": req.source,
        "data_provider": "Twelve Data",
        "candles_used": len(candles),
        "latest_completed_candle": str(candles[-1].time),
        "feed_meta": meta,
        **result,
    }
    save_history(record)

    # Do not return the API key.
    return record


@app.get("/api/health")
def health():
    return {"ok": True, "version": "V1", "engine": "11-indicator weighted scorer"}


@app.post("/api/signal")
def get_signal(req: SignalRequest):
    try:
        result = analyze(req.candles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    record = {
        "id": datetime.now(timezone.utc).isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "source": req.source,
        **result,
    }
    save_history(record)
    return record


@app.post("/api/candles")
def candle_input(req: CandleRequest):
    # Validates and accepts an OHLC batch. No market data is fabricated here.
    return {
        "accepted": len(req.candles),
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "last_candle": req.candles[-1].model_dump(),
        "message": "Candles accepted. POST the same payload to /api/signal for analysis.",
    }


@app.get("/api/history")
def history(limit: int = 50):
    return {"items": load_history()[:max(1, min(limit, 100))]}


@app.delete("/api/history")
def clear_history():
    HISTORY_FILE.write_text("[]")
    return {"ok": True}


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
