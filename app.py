from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, timezone

app = FastAPI()
FRED_KEY = os.getenv("FRED_API_KEY")


# ---------------------------
# HTTP helpers
# ---------------------------

def _safe_get_json(url, params=None, timeout=12):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------
# FRED helpers
# ---------------------------

def get_fred_series_window(series_id: str, n: int = 120) -> pd.Series:
    """
    Fetch last n observations from FRED.
    Returns pandas Series of floats indexed by date (ascending).
    """
    if not FRED_KEY:
        # Don’t crash the whole app if env var missing—return empty series
        return pd.Series(dtype=float)

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": n,
    }

    try:
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
        return pd.Series(df["value"].values, index=df["date"].values, dtype=float)

    except Exception:
        return pd.Series(dtype=float)


def series_stats(s: pd.Series):
    """last value, 1-step delta, 5-step delta"""
    if s is None or len(s) == 0:
        return {"last": np.nan, "d1": np.nan, "d5": np.nan}
    last = float(s.iloc[-1])
    d1 = float(last - s.iloc[-2]) if len(s) >= 2 else np.nan
    d5 = float(last - s.iloc[-6]) if len(s) >= 6 else np.nan
    return {"last": last, "d1": d1, "d5": d5}


def pct_change_series(s: pd.Series, n: int) -> float:
    if s is None or len(s) < (n + 1):
        return np.nan
    last = float(s.iloc[-1])
    prev = float(s.iloc[-1 - n])
    if prev == 0:
        return np.nan
    return (last / prev - 1.0) * 100.0


def sma_series(s: pd.Series, n: int) -> float:
    if s is None or len(s) < n:
        return np.nan
    return float(pd.Series(s).rolling(n).mean().iloc[-1])


def percentile_rank(window: pd.Series, x: float) -> float:
    """0..100 percentile rank of x vs window"""
    if window is None or len(window) == 0 or np.isnan(x):
        return np.nan
    w = pd.Series(window).dropna().values
    if len(w) == 0:
        return np.nan
    return float((w <= x).mean() * 100.0)


def atr_proxy_from_close(s: pd.Series, n: int) -> float:
    """
    FRED SP500/NASDAQCOM are closes only.
    Use average abs close-to-close change as a volatility proxy.
    """
    if s is None or len(s) < (n + 2):
        return np.nan
    diffs = pd.Series(s).diff().abs()
    return float(diffs.rolling(n).mean().iloc[-1])


# ---------------------------
# Coinbase crypto helpers
# ---------------------------

def get_coinbase_candles(product: str, start: datetime, end: datetime, granularity: int = 3600) -> pd.DataFrame:
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    params = {
        "granularity": granularity,
        "start": start.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "end": end.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        j = _safe_get_json(url, params=params, timeout=12)
        df = pd.DataFrame(j, columns=["time", "low", "high", "open", "close", "volume"])
        if df.empty:
            return df
        return df.sort_values("time")
    except Exception:
        return pd.DataFrame()


def crypto_return(product="BTC-USD", hours=24) -> float:
    """
    Percent return over last N hours using hourly candles.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours + 1)

    df = get_coinbase_candles(product, start=start, end=end, granularity=3600)
    if df is None or df.empty or len(df) < 2:
        return np.nan

    old = float(df.iloc[0]["close"])
    new = float(df.iloc[-1]["close"])
    if old == 0:
        return np.nan
    return (new - old) / old * 100.0


# ---------------------------
# Formatting helpers
# ---------------------------

def fmt_num(x, decimals=2, suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{decimals}f}{suffix}"


def fmt_delta(x, decimals=2, suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.{decimals}f}{suffix}"


# ---------------------------
# Interpretation helpers
# ---------------------------

def explain_indicator_blocks(vix, hy, y10, vix_pct, atr_expansion, momentum_accel, agree):
    why = []

    # VIX
    if not np.isnan(vix["last"]):
        if vix["last"] >= 25:
            why.append(f"VIX elevated ({fmt_num(vix['last'], 2)}) → markets pricing more fear/volatility.")
        elif vix["last"] >= 20:
            why.append(f"VIX moderately high ({fmt_num(vix['last'], 2)}) → more chop; leverage less forgiving.")
        else:
            why.append(f"VIX calm ({fmt_num(vix['last'], 2)}) → vol not screaming danger.")

    if not np.isnan(vix_pct):
        if vix_pct >= 80:
            why.append(f"VIX high vs ~90D (pctl {vix_pct:.0f}%) → higher whipsaw/decay risk.")
        elif vix_pct <= 30:
            why.append(f"VIX low vs ~90D (pctl {vix_pct:.0f}%) → friendlier for multi-day leverage.")

    # HY spreads
    if not np.isnan(hy["last"]):
        if hy["last"] >= 5.0:
            why.append(f"HY spreads wide ({fmt_num(hy['last'], 2)}) → credit stress rising; equities can follow lower.")
        elif hy["last"] >= 4.0:
            why.append(f"HY spreads drifting up ({fmt_num(hy['last'], 2)}) → mild risk-off from credit.")
        else:
            why.append(f"HY spreads contained ({fmt_num(hy['last'], 2)}) → credit not flashing red.")

    if not np.isnan(hy["d5"]) and hy["d5"] >= 0.25:
        why.append(f"HY widening over ~5D ({fmt_delta(hy['d5'], 2)}) → funding stress rising.")

    # 10Y
    if not np.isnan(y10["last"]):
        if y10["last"] >= 4.75:
            why.append(f"10Y high ({fmt_num(y10['last'], 2)}) → tighter conditions can pressure growth/tech.")
        elif y10["last"] >= 4.50:
            why.append(f"10Y elevated ({fmt_num(y10['last'], 2)}) → headwind if it keeps rising.")
        else:
            why.append(f"10Y not extreme ({fmt_num(y10['last'], 2)}) → rates not primary stressor.")

    # Vol expansion + momentum
    if atr_expansion:
        why.append("SPX volatility expanding (ATR proxy up) → more whipsaw; reduce leverage or expectations.")
    else:
        why.append("SPX volatility stable/compressed → better for multi-day leverage (if trend supports it).")

    if momentum_accel:
        why.append("NDX momentum accelerating (5D > 21D) → supports risk-on continuation.")
    else:
        why.append("NDX momentum not accelerating → expect chop/grind unless a catalyst hits.")

    # Crypto alignment
    if agree:
        why.append("BTC/ETH aligned → crypto signal quality higher today.")
    else:
        why.append("BTC/ETH diverge → crypto signal mixed today.")

    return why[:10]


def build_interpretation(light, confidence, leverage_regime, chips, why):
    if light == "GREEN":
        headline = "Trend beats stress (risk-on regime)."
        what = [
            "Best environment for multi-day leverage is: uptrend + contained stress + manageable volatility.",
            "Prefer buying pullbacks over chasing green candles."
        ]
        watchouts = [
            "If VIX jumps + HY spreads widen together, regime can flip quickly.",
            "Leverage decay still hurts in sideways chop."
        ]
    elif light == "YELLOW":
        headline = "Mixed regime (selective leverage)."
        what = [
            "Signals are not clean. Smaller size and confirmation matter more than being early.",
            "Consider partial hedges or shorter holds until stress gauges improve."
        ]
        watchouts = [
            "Chop weeks are where leveraged ETFs bleed quietly.",
            "If stress worsens, assume downside tails are larger."
        ]
    else:
        headline = "Stress beats trend (risk-off regime)."
        what = [
            "Defense first: cash/hedges beat forcing upside.",
            "Wait for stress to normalize + trend to recover before sizing up risk-on."
        ]
        watchouts = [
            "Bear market rallies are violent—don’t oversize inverses either.",
            "Avoid ‘revenge leverage’ after big down days."
        ]

    return {
        "headline": headline,
        "confidence": confidence,
        "leverage_regime": leverage_regime,
        "what_it_means": what,
        "watchouts": watchouts,
        "lever_menu": chips,
        "why": why,
        "disclaimer": "Educational only; not investment advice. Leveraged/inverse ETFs can lose rapidly, especially in volatile regimes."
    }


# ---------------------------
# Core: compute_stoplight
# ---------------------------

def compute_stoplight():
    # Pull data
    spx_s = get_fred_series_window("SP500", n=650)
    ndx_s = get_fred_series_window("NASDAQCOM", n=650)
    vix_s = get_fred_series_window("VIXCLS", n=120)
    y10_s = get_fred_series_window("DGS10", n=120)
    hy_s  = get_fred_series_window("BAMLH0A0HYM2", n=120)

    vix = series_stats(vix_s)
    y10 = series_stats(y10_s)
    hy  = series_stats(hy_s)

    # Index metrics
    spx_last = float(spx_s.iloc[-1]) if len(spx_s) else np.nan
    ndx_last = float(ndx_s.iloc[-1]) if len(ndx_s) else np.nan

    spx_5d  = pct_change_series(spx_s, 5)
    spx_21d = pct_change_series(spx_s, 21)
    ndx_5d  = pct_change_series(ndx_s, 5)
    ndx_21d = pct_change_series(ndx_s, 21)

    spx_50  = sma_series(spx_s, 50)
    spx_200 = sma_series(spx_s, 200)
    ndx_50  = sma_series(ndx_s, 50)
    ndx_200 = sma_series(ndx_s, 200)

    # Trend score
    trend_score = 0
    spx_above_50 = (not np.isnan(spx_last) and not np.isnan(spx_50) and spx_last > spx_50)
    spx_above_200 = (not np.isnan(spx_last) and not np.isnan(spx_200) and spx_last > spx_200)
    ndx_above_50 = (not np.isnan(ndx_last) and not np.isnan(ndx_50) and ndx_last > ndx_50)
    ndx_above_200 = (not np.isnan(ndx_last) and not np.isnan(ndx_200) and ndx_last > ndx_200)

    trend_score += 1 if spx_above_50 else -1
    trend_score += 1 if spx_above_200 else -1
    trend_score += 1 if ndx_above_50 else -1
    trend_score += 1 if ndx_above_200 else -1

    # New additions
    vix_90 = get_fred_series_window("VIXCLS", n=90)
    vix_pct = percentile_rank(vix_90, vix["last"])

    spx_atr10 = atr_proxy_from_close(spx_s, 10)
    spx_atr30 = atr_proxy_from_close(spx_s, 30)
    atr_expansion = False
    if not np.isnan(spx_atr10) and not np.isnan(spx_atr30) and spx_atr30 != 0:
        atr_expansion = spx_atr10 > spx_atr30 * 1.15

    momentum_accel = (not np.isnan(ndx_5d) and not np.isnan(ndx_21d) and ndx_5d > ndx_21d)

    # Crypto
    btc_24 = crypto_return("BTC-USD", 24)
    eth_24 = crypto_return("ETH-USD", 24)
    btc_7d = crypto_return("BTC-USD", 24 * 7)
    eth_7d = crypto_return("ETH-USD", 24 * 7)
    agree = (not np.isnan(btc_24) and not np.isnan(eth_24) and (btc_24 * eth_24) > 0)

    # Risk scoring
    risk = 0
    macro_flags = 0

    if not np.isnan(vix["last"]):
        if vix["last"] >= 30:
            risk += 3; macro_flags += 1
        elif vix["last"] >= 25:
            risk += 2; macro_flags += 1
        elif vix["last"] >= 20:
            risk += 1; macro_flags += 1

    if not np.isnan(vix["d5"]) and vix["d5"] >= 3.0:
        risk += 1
        macro_flags = max(macro_flags, 1)

    if not np.isnan(hy["last"]):
        if hy["last"] >= 6.0:
            risk += 3; macro_flags += 1
        elif hy["last"] >= 5.0:
            risk += 2; macro_flags += 1
        elif hy["last"] >= 4.0:
            risk += 1; macro_flags += 1

    if not np.isnan(hy["d5"]) and hy["d5"] >= 0.25:
        risk += 1
        macro_flags = max(macro_flags, 1)

    if not np.isnan(y10["last"]):
        if y10["last"] >= 4.75:
            risk += 2; macro_flags += 1
        elif y10["last"] >= 4.50:
            risk += 1; macro_flags += 1

    if not np.isnan(y10["d5"]) and y10["d5"] >= 0.25:
        risk += 1
        macro_flags = max(macro_flags, 1)

    if atr_expansion:
        risk += 1

    if not np.isnan(btc_24):
        if btc_24 <= -6.0:
            risk += 2
        elif btc_24 <= -3.0:
            risk += 1
        elif btc_24 >= 4.0:
            risk -= 1

    if not agree and (not np.isnan(btc_24) and not np.isnan(eth_24)):
        risk += 1

    net = trend_score - risk

    if net >= 2:
        light = "GREEN"
    elif net <= -2:
        light = "RED"
    else:
        light = "YELLOW"

    if macro_flags >= 3:
        confidence = "HIGH"
    elif macro_flags == 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    leverage_regime = "NEUTRAL"
    if (not np.isnan(vix_pct) and vix_pct <= 50) and (not atr_expansion) and momentum_accel and trend_score >= 2 and macro_flags == 0:
        leverage_regime = "FAVORABLE"
    if (not np.isnan(vix_pct) and vix_pct >= 75) or atr_expansion or macro_flags >= 2 or trend_score <= -2:
        leverage_regime = "UNFAVORABLE"

    if light == "GREEN":
        chips = ["TQQQ", "NVDL", "AMDL", "CONL", "BITX", "AGQ/JNUG (if metals also risk-on)"]
    elif light == "YELLOW":
        chips = ["Smaller size", "Shorter holds", "Prefer single-name leverage vs broad", "Partial hedge: PSQ/SQQQ", "Wait for confirmation"]
    else:
        chips = ["PSQ / SQQQ", "HIBS (aggressive hedge)", "Reduce/avoid leverage", "Cash is a position"]

    why = explain_indicator_blocks(vix, hy, y10, vix_pct, atr_expansion, momentum_accel, agree)
    interpretation = build_interpretation(light, confidence, leverage_regime, chips, why)

    metrics = [
        {"name": "BTC", "value": f"{fmt_num(btc_24, 2)}% (24h)", "delta": f"{fmt_num(btc_7d, 2)}% (7d)"},
        {"name": "ETH", "value": f"{fmt_num(eth_24, 2)}% (24h)", "delta": f"{fmt_num(eth_7d, 2)}% (7d)"},
        {"name": "VIX", "value": fmt_num(vix["last"], 2), "delta": f"5D {fmt_delta(vix['d5'], 2)} | Pctl90D {fmt_num(vix_pct, 0)}"},
        {"name": "10Y", "value": fmt_num(y10["last"], 2), "delta": f"5D {fmt_delta(y10['d5'], 2)}"},
        {"name": "HY Spread", "value": fmt_num(hy["last"], 2), "delta": f"5D {fmt_delta(hy['d5'], 2)}"},
        {"name": "SPX", "value": fmt_num(spx_last, 2), "delta": f"5D {fmt_num(spx_5d, 2)}% | 21D {fmt_num(spx_21d, 2)}%"},
        {"name": "NDX", "value": fmt_num(ndx_last, 2), "delta": f"5D {fmt_num(ndx_5d, 2)}% | 21D {fmt_num(ndx_21d, 2)}%"},
        {"name": "Trend score", "value": str(trend_score), "delta": f"SPX>50/200: {spx_above_50}/{spx_above_200} | NDX>50/200: {ndx_above_50}/{ndx_above_200}"},
        {"name": "Leverage regime", "value": leverage_regime, "delta": f"ATR exp: {atr_expansion} | Mom accel: {momentum_accel}"},
        {"name": "Net", "value": str(net), "delta": f"Risk: {risk} | Macro flags: {macro_flags} | Crypto agree: {agree}"},
    ]

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "light": light,
        "confidence": confidence,
        "net": int(net),
        "risk_points": int(risk),
        "trend_score": int(trend_score),
        "macro_flags": int(macro_flags),
        "leverage_regime": leverage_regime,
        "metrics": metrics,
        "interpretation": interpretation,
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
    light = data["light"]
    color = {"GREEN": "#2ee59d", "YELLOW": "#ffd166", "RED": "#ff5c7a"}[light]
    interp = data["interpretation"]

    cards_html = "".join([
        f"""
        <div class="card">
          <div class="k">{m["name"]}</div>
          <div class="row">
            <div class="val">{m["value"]}</div>
            <div class="delta">{m["delta"]}</div>
          </div>
        </div>
        """ for m in data["metrics"]
    ])

    why_html = "".join([f"<li>{x}</li>" for x in interp["why"]])
    menu_html = "".join([f'<span class="chip">{x}</span>' for x in interp["lever_menu"]])
    what_html = "".join([f"<li>{x}</li>" for x in interp["what_it_means"]])
    watch_html = "".join([f"<li>{x}</li>" for x in interp["watchouts"]])

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Market Stoplight</title>
  <style>
    :root {{
      --bg:#070a12;
      --panel:rgba(255,255,255,.06);
      --border:rgba(255,255,255,.12);
      --muted:rgba(255,255,255,.72);
      --muted2:rgba(255,255,255,.55);
      --accent:{color};
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
    }}
    body {{ margin:0; font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial; background:var(--bg); color:#fff; }}
    .wrap {{ max-width: 860px; margin:0 auto; padding:22px 16px 40px; }}
    .top {{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px; align-items:baseline; }}
    .title {{ font-size: 22px; font-weight: 800; }}
