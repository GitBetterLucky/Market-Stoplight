from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, timezone

app = FastAPI()

FRED_KEY = os.getenv("FRED_API_KEY")  # set this in Render env vars


# ---------------------------
# Data helpers
# ---------------------------

def _safe_get_json(url, params=None, timeout=12):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_fred_series_window(series_id: str, n: int = 20) -> pd.Series:
    """
    Fetch last n observations from FRED.
    Returns a pandas Series of floats indexed by date string (oldest->newest).
    """
    if not FRED_KEY:
        raise RuntimeError("Missing FRED_API_KEY environment variable")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": n,
    }
    j = _safe_get_json(url, params=params)
    obs = j.get("observations", [])
    if not obs:
        return pd.Series(dtype=float)

    df = pd.DataFrame(obs)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        return pd.Series(dtype=float)

    df = df.sort_values("date")  # oldest -> newest
    return pd.Series(df["value"].values, index=df["date"].values)

def get_fred_series_df(series_id: str, limit: int = 600) -> pd.DataFrame:
    """
    Fetch last `limit` observations from FRED as a dataframe with columns: date, value (float).
    Oldest->newest.
    """
    if not FRED_KEY:
        raise RuntimeError("Missing FRED_API_KEY environment variable")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    j = _safe_get_json(url, params=params)
    obs = j.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.sort_values("date")  # oldest -> newest
    return df[["date", "value"]].reset_index(drop=True)

def series_stats(s: pd.Series):
    """
    last value, 1-step delta, 5-step delta (approx 1 week trading days)
    """
    if s is None or len(s) == 0:
        return {"last": np.nan, "d1": np.nan, "d5": np.nan}

    last = float(s.iloc[-1])
    d1 = float(last - s.iloc[-2]) if len(s) >= 2 else np.nan
    d5 = float(last - s.iloc[-6]) if len(s) >= 6 else np.nan
    return {"last": last, "d1": d1, "d5": d5}

def pct_change(df: pd.DataFrame, lag: int = 1) -> float:
    """
    Percent change between last value and value `lag` steps back.
    """
    if df is None or df.empty or len(df) <= lag:
        return np.nan
    last = float(df["value"].iloc[-1])
    prev = float(df["value"].iloc[-(lag + 1)])
    if prev == 0:
        return np.nan
    return (last - prev) / prev * 100.0

def sma(df: pd.DataFrame, window: int) -> float:
    """
    Simple moving average of last `window` observations (returns last SMA value).
    """
    if df is None or df.empty or len(df) < window:
        return np.nan
    return float(pd.Series(df["value"]).rolling(window).mean().iloc[-1])

def get_coinbase_candles(product: str, start: datetime, end: datetime, granularity: int = 3600) -> pd.DataFrame:
    """
    Coinbase Exchange candles endpoint supports start/end ISO8601.
    Returns DF sorted by time ascending.
    """
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    params = {
        "granularity": granularity,
        "start": start.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "end": end.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    j = _safe_get_json(url, params=params)
    df = pd.DataFrame(j, columns=["time", "low", "high", "open", "close", "volume"])
    if df.empty:
        return df
    return df.sort_values("time")

def crypto_return(product="BTC-USD", hours=24) -> float:
    """
    Percent return over last N hours using hourly candles.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours + 1)

    df = get_coinbase_candles(product, start=start, end=end, granularity=3600)
    if df is None or df.empty or len(df) < 2:
        return 0.0

    old = float(df.iloc[0]["close"])
    new = float(df.iloc[-1]["close"])
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0

def fmt_num(x, decimals=2, suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{decimals}f}{suffix}"

def fmt_delta(x, decimals=2, suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.{decimals}f}{suffix}"


# ---------------------------
# Stoplight logic
# ---------------------------

def compute_stoplight():
    # --- pull crypto impulse ---
    btc_24 = crypto_return("BTC-USD", 24)
    eth_24 = crypto_return("ETH-USD", 24)
    btc_7d = crypto_return("BTC-USD", 24 * 7)
    eth_7d = crypto_return("ETH-USD", 24 * 7)

    # --- pull macro stress levers from FRED ---
    vix_s = get_fred_series_window("VIXCLS", n=30)
    hy_s  = get_fred_series_window("BAMLH0A0HYM2", n=30)
    y10_s = get_fred_series_window("DGS10", n=30)

    vix = series_stats(vix_s)
    hy  = series_stats(hy_s)
    y10 = series_stats(y10_s)

    # --- pull equity tape for trend regime ---
    spx = get_fred_series_df("SP500", limit=600)
    ndx = get_fred_series_df("NASDAQCOM", limit=600)

    spx_last = float(spx["value"].iloc[-1]) if not spx.empty else np.nan
    ndx_last = float(ndx["value"].iloc[-1]) if not ndx.empty else np.nan

    spx_1d = pct_change(spx, 1)
    spx_5d = pct_change(spx, 5)
    spx_21d = pct_change(spx, 21)

    ndx_1d = pct_change(ndx, 1)
    ndx_5d = pct_change(ndx, 5)
    ndx_21d = pct_change(ndx, 21)

    spx_50 = sma(spx, 50)
    spx_200 = sma(spx, 200)
    ndx_50 = sma(ndx, 50)
    ndx_200 = sma(ndx, 200)

    # Trend score (+ supports risk-on; - supports risk-off)
    trend_score = 0
    if not np.isnan(spx_last) and not np.isnan(spx_50):
        trend_score += 1 if spx_last > spx_50 else -1
    if not np.isnan(spx_last) and not np.isnan(spx_200):
        trend_score += 1 if spx_last > spx_200 else -1
    if not np.isnan(ndx_last) and not np.isnan(ndx_50):
        trend_score += 1 if ndx_last > ndx_50 else -1
    if not np.isnan(ndx_last) and not np.isnan(ndx_200):
        trend_score += 1 if ndx_last > ndx_200 else -1

    # Risk scoring (your existing framework)
    risk = 0
    why = []

    # VIX level + jump
    if not np.isnan(vix["last"]):
        if vix["last"] >= 25:
            risk += 2
            why.append(f"VIX elevated ({fmt_num(vix['last'], 2)}).")
        elif vix["last"] >= 20:
            risk += 1
            why.append(f"VIX moderately high ({fmt_num(vix['last'], 2)}).")

    if not np.isnan(vix["d5"]) and vix["d5"] >= 3.0:
        risk += 1
        why.append(f"VIX rising over ~5D ({fmt_delta(vix['d5'], 2)}).")

    # HY spread level + widening
    if not np.isnan(hy["last"]):
        if hy["last"] >= 5.0:
            risk += 2
            why.append(f"High-yield spreads wide ({fmt_num(hy['last'], 2)}).")
        elif hy["last"] >= 4.0:
            risk += 1
            why.append(f"HY spreads drifting higher ({fmt_num(hy['last'], 2)}).")

    if not np.isnan(hy["d5"]) and hy["d5"] >= 0.25:
        risk += 1
        why.append(f"HY spreads widening over ~5D ({fmt_delta(hy['d5'], 2)}).")

    # 10Y yield shock (fast up moves tighten conditions)
    if not np.isnan(y10["last"]):
        if y10["last"] >= 4.75:
            risk += 2
            why.append(f"10Y yield high ({fmt_num(y10['last'], 2)}).")
        elif y10["last"] >= 4.50:
            risk += 1
            why.append(f"10Y yield elevated ({fmt_num(y10['last'], 2)}).")

    if not np.isnan(y10["d5"]) and y10["d5"] >= 0.25:
        risk += 1
        why.append(f"10Y moved up fast over ~5D ({fmt_delta(y10['d5'], 2)}).")

    # Crypto stress (your idea: crypto as risk appetite / stress proxy)
    if btc_24 <= -3.0:
        risk += 1
        why.append(f"BTC down hard 24h ({fmt_num(btc_24, 2, '%')}).")
    if btc_24 <= -6.0:
        risk += 1
        why.append("BTC drawdown is severe (risk-off impulse).")

    if btc_24 >= 4.0:
        risk -= 1
        why.append(f"BTC strong 24h ({fmt_num(btc_24, 2, '%')}) — risk appetite improving.")

    agree = (btc_24 * eth_24) > 0
    why.append("BTC/ETH agree on direction." if agree else "BTC/ETH diverge (lower signal quality).")

    # Use trend to offset (or add) risk a bit
    net_risk = risk - trend_score  # trend_score positive reduces risk; negative increases it

    # Regime mapping (based on net risk)
    if net_risk <= 0:
        light = "GREEN"
    elif net_risk <= 3:
        light = "YELLOW"
    else:
        light = "RED"

    # Confidence: how many independent “macro” levers are flashing risk?
    macro_flags = 0
    if (not np.isnan(vix["last"]) and vix["last"] >= 20) or (not np.isnan(vix["d5"]) and vix["d5"] >= 3.0):
        macro_flags += 1
    if (not np.isnan(hy["last"]) and hy["last"] >= 4.0) or (not np.isnan(hy["d5"]) and hy["d5"] >= 0.25):
        macro_flags += 1
    if (not np.isnan(y10["last"]) and y10["last"] >= 4.50) or (not np.isnan(y10["d5"]) and y10["d5"] >= 0.25):
        macro_flags += 1

    confidence = "HIGH" if macro_flags >= 3 else ("MEDIUM" if macro_flags == 2 else "LOW")

    # “lever menu” (not advice, just quick mapping)
    if light == "GREEN":
        chips = ["TQQQ", "NVDL", "AMDL", "CONL", "BITX", "AGQ (if metals risk-on)"]
    elif light == "YELLOW":
        chips = ["Smaller size", "Shorter holds", "Prefer NVDL/AMDL over TQQQ", "Pair with partial hedge: PSQ/SQQQ", "Wait for confirmation"]
    else:
        chips = ["SQQQ / PSQ", "HIBS (aggressive hedge)", "Reduce leverage", "Cash is a position"]

    metrics = [
        {"name": "BTC 24h", "value": f"{fmt_num(btc_24, 2)}%", "delta": f"7D {fmt_num(btc_7d, 2)}%"},
        {"name": "ETH 24h", "value": f"{fmt_num(eth_24, 2)}%", "delta": f"7D {fmt_num(eth_7d, 2)}%"},
        {"name": "VIX", "value": fmt_num(vix["last"], 2), "delta": f"1D {fmt_delta(vix['d1'], 2)} | 5D {fmt_delta(vix['d5'], 2)}"},
        {"name": "10Y", "value": fmt_num(y10["last"], 2), "delta": f"1D {fmt_delta(y10['d1'], 2)} | 5D {fmt_delta(y10['d5'], 2)}"},
        {"name": "HY Spread", "value": fmt_num(hy["last"], 2), "delta": f"1D {fmt_delta(hy['d1'], 2)} | 5D {fmt_delta(hy['d5'], 2)}"},
        {"name": "SPX", "value": fmt_num(spx_last, 2), "delta": f"5D {fmt_delta(spx_5d, 2, '%')} | 21D {fmt_delta(spx_21d, 2, '%')}"},
        {"name": "NDX", "value": fmt_num(ndx_last, 2), "delta": f"5D {fmt_delta(ndx_5d, 2, '%')} | 21D {fmt_delta(ndx_21d, 2, '%')}"},
        {"name": "Trend score", "value": str(int(trend_score)), "delta": f"SPX>50/200: {spx_last > spx_50 if not np.isnan(spx_50) else 'NA'}/{spx_last > spx_200 if not np.isnan(spx_200) else 'NA'}"},
        {"name": "Net risk", "value": str(int(net_risk)), "delta": f"Raw risk: {risk} | Macro flags: {macro_flags} | Signal: {'Agree' if agree else 'Mixed'}"},
    ]

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "light": light,
        "risk_points": int(net_risk),
        "confidence": confidence,
        "btc_24h": round(btc_24, 2),
        "eth_24h": round(eth_24, 2),
        "btc_7d": round(btc_7d, 2),
        "eth_7d": round(eth_7d, 2),
        "vix": vix,
        "y10": y10,
        "hy_spread": hy,
        "why": why[:8],
        "chips": chips,
        "metrics": metrics,
    }


# ---------------------------
# Routes
# ---------------------------

@app.get("/status")
def status():
    return compute_stoplight()

@app.get("/", response_class=HTMLResponse)
def homepage():
    data = compute_stoplight()
    color = {"GREEN": "#0f0", "YELLOW": "#ff0", "RED": "#f00"}[data["light"]]

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Market Stoplight</title>
  <style>
    body {{ margin:0; font-family: Arial, sans-serif; background:#000; color:#fff; }}
    .wrap {{ max-width: 520px; margin: 0 auto; padding: 22px 16px 30px; text-align:center; }}
    .light {{ width: 220px; height: 220px; border-radius: 50%; margin: 18px auto; background: {color};
             box-shadow: 0 0 40px rgba(255,255,255,0.08); }}
    h1 {{ margin: 0 0 6px 0; }}
    .label {{ font-size: 28px; font-weight: 700; margin-top: 6px; }}
    .sub {{ opacity: 0.8; margin-top: 6px; }}
    .cards {{ text-align:left; margin-top: 18px; }}
    .card {{ border: 1px solid rgba(255,255,255,0.15); border-radius: 14px; padding: 12px 14px; margin-bottom: 10px; }}
    .k {{ opacity: 0.75; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
    .v {{ font-size: 18px; margin-top: 4px; display:flex; justify-content:space-between; gap: 10px; }}
    .delta {{ opacity: 0.8; font-size: 14px; text-align:right; }}
    .why {{ margin-top: 14px; text-align:left; }}
    .why ul {{ margin: 8px 0 0 18px; }}
    .chips {{ margin-top: 14px; text-align:left; }}
    .chip {{ display:inline-block; padding: 6px 10px; border-radius: 999px;
             border: 1px solid rgba(255,255,255,0.18); margin: 6px 6px 0 0; font-size: 13px; }}
    .foot {{ opacity:.6; font-size:12px; margin-top:18px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Market Stoplight</h1>
    <div class="light"></div>
    <div class="label">{data["light"]}</div>
    <div class="sub">Confidence: {data["confidence"]} · Net risk: {data["risk_points"]}</div>

    <div class="cards">
      {''.join([f'''
      <div class="card">
        <div class="k">{m["name"]}</div>
        <div class="v">
          <div>{m["value"]}</div>
          <div class="delta">{m["delta"]}</div>
        </div>
      </div>
      ''' for m in data["metrics"]])}
    </div>

    <div class="why">
      <div class="k">Why</div>
      <ul>
        {''.join([f"<li>{w}</li>" for w in data["why"]])}
      </ul>
    </div>

    <div class="chips">
      <div class="k">Levers (menu, not advice)</div>
      {''.join([f'<span class="chip">{c}</span>' for c in data["chips"]])}
    </div>

    <div class="foot">Updated: {data["timestamp"]}</div>
  </div>
</body>
</html>
"""
