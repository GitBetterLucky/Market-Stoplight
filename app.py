from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
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
# Regime 2.0 — pillar scoring (swing-first) + tape overlay
# ---------------------------

def _sgn(x):
    if np.isnan(x):
        return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)

def _banded_vs_zero(x, pos=0.30, neg=-0.30):
    """Percent-change style inputs. Returns +1 / 0 / -1 with a small noise band."""
    if np.isnan(x):
        return 0
    if x >= pos:
        return 1
    if x <= neg:
        return -1
    return 0

def _ratio_trend(r, band=0.003):
    """
    Ratios like RSP/SPY, IWM/SPY, HYG/LQD:
    We don't have a ref series yet, so we infer trend from same-day relative moves:
    - If numerator pct > denom pct by > band*100, call it UP
    - Else DOWN if < -band*100, else FLAT
    """
    return r  # placeholder if you later add MA/ref; keep function for evolution

def regime_2_0_scores(
    # trend
    spx_above_50, spx_above_200,
    ndx_above_50, ndx_above_200,
    dji_above_50, dji_above_200,

    # stress (levels + momentum)
    vix_last, vix_pct, hy_spread_last, hy_spread_d5, y10_last, atr_expansion,

    # cross-asset pct
    spy_pct, qqq_pct, dia_pct,
    rsp_pct, iwm_pct,
    hyg_pct, lqd_pct,
    uup_pct, gld_pct, tlt_pct, uso_pct,

    # overlays
    tape_avg,
):
    scores = {
        "trend": 0,
        "participation": 0,
        "liquidity": 0,
        "stress": 0,
        "tape": 0,
    }
    notes = []

    # ---------------------------
    # Trend pillar (slow)
    # ---------------------------
    # 50d = tactical, 200d = structural
    scores["trend"] += (1 if spx_above_50 else -1)
    scores["trend"] += (1 if spx_above_200 else -1)

    scores["trend"] += (1 if ndx_above_50 else -1)
    scores["trend"] += (1 if ndx_above_200 else -1)

    scores["trend"] += (1 if dji_above_50 else -1)
    scores["trend"] += (1 if dji_above_200 else -1)

    # normalize to a tighter band (-3..+3)
    # (optional) compress: take sign of each index block
    # leaving as-is for now because you already display trend_score

    # ---------------------------
    # Participation pillar (breadth/leadership)
    # ---------------------------
    # RSP vs SPY (broad market participation)
    if not np.isnan(rsp_pct) and not np.isnan(spy_pct):
        if (rsp_pct - spy_pct) >= 0.20:
            scores["participation"] += 1
            notes.append("Breadth: equal-weight leading (healthy participation).")
        elif (rsp_pct - spy_pct) <= -0.20:
            scores["participation"] -= 1
            notes.append("Breadth: cap-weight leading (narrow leadership).")

    # IWM vs SPY (risk appetite / liquidity)
    if not np.isnan(iwm_pct) and not np.isnan(spy_pct):
        if (iwm_pct - spy_pct) >= 0.20:
            scores["participation"] += 1
            notes.append("Leadership: small caps leading (risk appetite improving).")
        elif (iwm_pct - spy_pct) <= -0.20:
            scores["participation"] -= 1
            notes.append("Leadership: small caps lagging (risk appetite fading).")

    # ---------------------------
    # Stress pillar (fear + funding + rates + vol expansion)
    # ---------------------------
    # VIX level & percentile
    if not np.isnan(vix_last):
        if vix_last >= 25:
            scores["stress"] -= 2
        elif vix_last >= 20:
            scores["stress"] -= 1
        elif vix_last <= 14:
            scores["stress"] += 1

    if not np.isnan(vix_pct):
        if vix_pct >= 80:
            scores["stress"] -= 1
        elif vix_pct <= 30:
            scores["stress"] += 1

    # HY spread level + widening
    if not np.isnan(hy_spread_last):
        if hy_spread_last >= 5.0:
            scores["stress"] -= 2
        elif hy_spread_last >= 4.0:
            scores["stress"] -= 1
        elif hy_spread_last <= 3.5:
            scores["stress"] += 1

    if not np.isnan(hy_spread_d5) and hy_spread_d5 >= 0.25:
        scores["stress"] -= 1

    # 10Y too high pressures growth
    if not np.isnan(y10_last):
        if y10_last >= 4.75:
            scores["stress"] -= 1
        elif y10_last <= 3.75:
            scores["stress"] += 1

    # ATR expansion = choppier / worse for leverage
    if atr_expansion:
        scores["stress"] -= 1

    # Credit tape (HYG vs LQD) — directional early-warning
    if not np.isnan(hyg_pct) and not np.isnan(lqd_pct):
        if hyg_pct < 0 and lqd_pct >= 0:
            scores["stress"] -= 1
            notes.append("Credit: high yield weaker than IG (funding stress creeping in).")
        elif hyg_pct > 0 and lqd_pct <= 0:
            scores["stress"] += 1
            notes.append("Credit: high yield stronger than IG (risk appetite supported).")

    # ---------------------------
    # Liquidity / hedge pillar (USD, bonds, gold, oil as context)
    # ---------------------------
    # Strong USD up + stocks down often = tightening
    if not np.isnan(uup_pct) and not np.isnan(spy_pct):
        if uup_pct >= 0.50 and spy_pct < 0:
            scores["liquidity"] -= 1
            notes.append("Liquidity: dollar up while stocks down (tightening signal).")
        elif uup_pct <= -0.50 and spy_pct > 0:
            scores["liquidity"] += 1
            notes.append("Liquidity: dollar down while stocks up (easing tailwind).")

    # Flight-to-safety confirmation: SPY down + TLT up
    if not np.isnan(tlt_pct) and not np.isnan(spy_pct):
        if spy_pct < -0.30 and tlt_pct > 0.30:
            scores["liquidity"] -= 1
            notes.append("Positioning: bonds bid on equity weakness (defensive posture).")

    # Gold up sharply on equity weakness = risk-off hedge demand (soft confirm)
    if not np.isnan(gld_pct) and not np.isnan(spy_pct):
        if spy_pct < -0.30 and gld_pct > 0.30:
            scores["liquidity"] -= 1

    # Oil sharp down can be growth scare (context only for now)
    # (keep as note, not scoring, unless you want it to matter)
    if not np.isnan(uso_pct) and uso_pct <= -1.0:
        notes.append("Macro context: oil down hard (watch for growth-scare narratives).")

    # ---------------------------
    # Tape overlay (fast)
    # ---------------------------
    # This should not dominate swing regime, but should block leverage on ugly days.
    if not np.isnan(tape_avg):
        if tape_avg <= -0.80:
            scores["tape"] -= 2
            notes.append("Tape: broad index tape is risk-off today (avoid long leverage).")
        elif tape_avg >= 0.80:
            scores["tape"] += 1
            notes.append("Tape: broad index tape is strong today (risk-on follow-through more likely).")

    return scores, notes

def regime_2_0_net(scores):
    """
    Swing-first weights.
    Tape is an overlay: it can downgrade, but shouldn't overpower a healthy swing regime.
    """
    trend = scores.get("trend", 0)
    participation = scores.get("participation", 0)
    stress = scores.get("stress", 0)
    liquidity = scores.get("liquidity", 0)
    tape = scores.get("tape", 0)

    # weights
    net = (
        0.45 * trend +
        0.20 * participation +
        0.20 * (liquidity) +
        0.35 * (stress) +
        0.30 * (tape)
    )
    return int(round(net))

def stoplight_5_tier(net):
    if net <= -5:
        return "RED", "HIGH RISK", "🛑"
    if net <= -2:
        return "ORANGE", "ELEVATED RISK", "⚠️"
    if net <= 1:
        return "YELLOW", "NEUTRAL", "🟡"
    if net <= 4:
        return "GREEN", "LOW RISK", "🟢"
    return "STAR", "NIRVANA", "⭐"
    
# ---------------------------
# FRED helpers
# ---------------------------

def get_fred_series_window(series_id: str, n: int = 120) -> pd.Series:
    """
    Fetch last n observations from FRED.
    Returns pandas Series of floats indexed by date (ascending).
    """
    if not FRED_KEY:
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

def _yn(flag: bool):
    return "Above" if flag else "Below"

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
# Cross-asset + breadth helpers (Yahoo) — no key
# ---------------------------

def pick_last(quote):
    """Return regularMarketPrice as float or NaN."""
    try:
        if not quote:
            return np.nan
        v = quote.get("regularMarketPrice", None)
        return float(v) if v is not None else np.nan
    except Exception:
        return np.nan

def ratio_momentum(a_last, b_last):
    """
    Simple ratio (a/b). Returns NaN if missing.
    We’ll use this for proxies like RSP/SPY, HYG/LQD, IWM/SPY.
    """
    try:
        if np.isnan(a_last) or np.isnan(b_last) or b_last == 0:
            return np.nan
        return float(a_last / b_last)
    except Exception:
        return np.nan

def classify_ratio(r_now, r_ref, band=0.003):
    """
    Compare ratio now vs reference (e.g., yesterday or a slow MA later).
    band ~0.3% default. Returns: 'UP', 'DOWN', 'FLAT', or 'NA'
    """
    if np.isnan(r_now) or np.isnan(r_ref) or r_ref == 0:
        return "NA"
    chg = (r_now / r_ref) - 1.0
    if chg > band:
        return "UP"
    if chg < -band:
        return "DOWN"
    return "FLAT"

def yf_quotes(symbols):
    """
    Batch fetch last + prev close using yfinance download.
    Returns dict: {SYM: {"regularMarketPrice": last_close, "regularMarketPreviousClose": prev_close}}
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    if not symbols:
        return {}

    out = {s.upper(): None for s in symbols}

    try:
        data = yf.download(
            tickers=symbols,
            period="2d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        def last_two(sym: str):
            if isinstance(data.columns, pd.MultiIndex):
                if (sym, "Close") not in data.columns:
                    return (np.nan, np.nan)
                closes = data[(sym, "Close")].dropna()
            else:
                closes = data["Close"].dropna()
            if len(closes) == 0:
                return (np.nan, np.nan)
            if len(closes) == 1:
                return (float(closes.iloc[-1]), np.nan)
            return (float(closes.iloc[-1]), float(closes.iloc[-2]))

        for sym in symbols:
            s = sym.upper()
            last_close, prev_close = last_two(s)
            if not np.isnan(last_close):
                out[s] = {
                    "regularMarketPrice": last_close,
                    "regularMarketPreviousClose": prev_close if not np.isnan(prev_close) else None,
                }
        return out

    except Exception:
        return out

def pick_pct_from_price(quote):
    try:
        if not quote:
            return np.nan
        last = quote.get("regularMarketPrice", None)
        prev = quote.get("regularMarketPreviousClose", None)
        if last is None or prev in (None, 0):
            return np.nan
        return (float(last) / float(prev) - 1.0) * 100.0
    except Exception:
        return np.nan

def cross_asset_snapshot():
    """
    Pulls best-effort cross-asset proxies from Yahoo.
    Returns dict with pct changes and ratio states.
    """
    syms = ["SPY","QQQ","DIA","RSP","IWM","HYG","LQD","UUP","GLD","USO","TLT","JPY=X"]
    q = yf_quotes(syms)

    # live pct (premarket if present, else regular)
    pct = {s: pick_pct_from_price(q.get(s)) for s in syms}
    last = {s: pick_last(q.get(s)) for s in syms}

    # ratios (instant snapshot)
    rsp_spy = ratio_momentum(last.get("RSP", np.nan), last.get("SPY", np.nan))
    iwm_spy = ratio_momentum(last.get("IWM", np.nan), last.get("SPY", np.nan))
    hyg_lqd = ratio_momentum(last.get("HYG", np.nan), last.get("LQD", np.nan))

    return {
        "pct": pct,
        "last": last,
        "ratios": {
            "RSP/SPY": rsp_spy,
            "IWM/SPY": iwm_spy,
            "HYG/LQD": hyg_lqd,
        }
    }

def tape_signal_from_pct(pct: dict):
    """
    Tape score using already-fetched cross-asset pct dict.
    Avoids double Yahoo pull.
    """
    spy = pct.get("SPY", np.nan)
    qqq = pct.get("QQQ", np.nan)
    dia = pct.get("DIA", np.nan)

    vals = [x for x in [spy, qqq, dia] if not np.isnan(x)]
    avg = float(np.mean(vals)) if vals else np.nan

    if not np.isnan(avg) and avg <= -0.80:
        bias = "BEAR"
    elif not np.isnan(avg) and avg >= 0.80:
        bias = "BULL"
    else:
        bias = "NEUTRAL"

    return {"spy": spy, "qqq": qqq, "dia": dia, "avg": avg, "bias": bias}

# ---------------------------
# Historical sector backtest (A)
# ---------------------------

SECTOR_UNIVERSE = {
    # broad / benchmark
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",

    # defensives
    "XLP": "Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",

    # cyclicals / rate sensitive
    "XLF": "Financials",
    "XLI": "Industrials",

    # geopolitics / shock buckets
    "XLE": "Energy",
    "ITA": "Aerospace & Defense",
    "GLD": "Gold",
    "SLV": "Silver",

    # shipping proxy (imperfect, but common)
    "BDRY": "Shipping (Dry Bulk)",
}

def _to_trading_day(index: pd.DatetimeIndex, dt: pd.Timestamp) -> pd.Timestamp | None:
    """Map a calendar date to the next available trading day in the price index."""
    if dt in index:
        return dt
    # next trading day
    pos = index.searchsorted(dt)
    if pos >= len(index):
        return None
    return index[pos]

def yf_adjclose(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download Adj Close for tickers between start and end (YYYY-MM-DD).
    Returns a DataFrame indexed by trading day.
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="ticker",
    )
    # Normalize to Adj Close DataFrame
    if isinstance(data.columns, pd.MultiIndex):
        out = {}
        for t in tickers:
            if (t, "Adj Close") in data.columns:
                out[t] = data[(t, "Adj Close")]
            elif (t, "Close") in data.columns:
                out[t] = data[(t, "Close")]
        df = pd.DataFrame(out)
    else:
        # single ticker shape
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        df = pd.DataFrame({tickers[0]: data[col]})
    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def compute_forward_returns(
    prices: pd.DataFrame,
    event_dates: list[str],
    horizons=(1, 5, 21),
) -> pd.DataFrame:
    """
    prices: DataFrame of adj close prices, columns=tickers, index=trading days
    event_dates: list of 'YYYY-MM-DD' strings (calendar dates; will snap to next trading day)
    Returns a long DataFrame with rows = (event, ticker), cols = horizon returns (%).
    """
    px = prices.copy()
    px = px.sort_index()
    idx = px.index

    # Precompute forward return matrices per horizon
    fwd = {}
    for h in horizons:
        fwd[h] = (px.shift(-h) / px - 1.0) * 100.0

    rows = []
    for d in event_dates:
        dt = pd.Timestamp(d)
        tday = _to_trading_day(idx, dt)
        if tday is None:
            continue

        for ticker in px.columns:
            rec = {"event_date": d, "tday": tday.date().isoformat(), "ticker": ticker}
            for h in horizons:
                val = fwd[h].loc[tday, ticker] if (tday in fwd[h].index) else np.nan
                rec[f"r{h}d"] = float(val) if pd.notna(val) else np.nan
            rows.append(rec)

    return pd.DataFrame(rows)

def summarize_forward_returns(
    fwd_long: pd.DataFrame,
    benchmark: str = "SPY",
    horizons=(1, 5, 21),
) -> dict:
    """
    Produces per-ticker summary stats:
      - median/mean return by horizon
      - hit-rate (% positive)
      - n (non-NA samples)
      - excess vs benchmark (median)
    """
    if fwd_long is None or fwd_long.empty:
        return {"error": "No forward returns computed", "tickers": []}

    out = {"benchmark": benchmark, "tickers": []}

    # benchmark medians for "excess" comparisons
    bench = fwd_long[fwd_long["ticker"] == benchmark]
    bench_median = {}
    for h in horizons:
        col = f"r{h}d"
        bench_median[h] = float(np.nanmedian(bench[col].values)) if not bench.empty else np.nan

    for ticker, g in fwd_long.groupby("ticker"):
        item = {"ticker": ticker}
        for h in horizons:
            col = f"r{h}d"
            vals = g[col].values.astype(float)
            n = int(np.isfinite(vals).sum())
            med = float(np.nanmedian(vals)) if n else np.nan
            mean = float(np.nanmean(vals)) if n else np.nan
            hit = float((vals[np.isfinite(vals)] > 0).mean() * 100.0) if n else np.nan
            item[f"{h}d_median"] = med
            item[f"{h}d_mean"] = mean
            item[f"{h}d_hit_rate"] = hit
            item[f"{h}d_n"] = n
            # excess median vs benchmark
            bm = bench_median.get(h, np.nan)
            item[f"{h}d_excess_median_vs_{benchmark}"] = (med - bm) if (np.isfinite(med) and np.isfinite(bm)) else np.nan

        out["tickers"].append(item)

    # Sort: best “shock winners” by 5D excess median, then 21D
    def _key(x):
        a = x.get(f"5d_excess_median_vs_{benchmark}", np.nan)
        b = x.get(f"21d_excess_median_vs_{benchmark}", np.nan)
        a = -1e9 if not np.isfinite(a) else a
        b = -1e9 if not np.isfinite(b) else b
        return (a, b)

    out["tickers"] = sorted(out["tickers"], key=_key, reverse=True)
    return out

def backtest_sectors_on_events(
    event_dates: list[str],
    universe: dict = SECTOR_UNIVERSE,
    start: str | None = None,
    end: str | None = None,
    horizons=(1, 5, 21),
    benchmark="SPY",
) -> dict:
    """
    Convenience wrapper: downloads prices + computes forward returns + summarizes.
    """
    tickers = list(universe.keys())

    # Default start/end: pad enough to compute 21D forward returns
    if start is None:
        # go back 10 years by default (you can tighten later)
        start = (datetime.now() - timedelta(days=365 * 10)).date().isoformat()
    if end is None:
        end = (datetime.now() + timedelta(days=10)).date().isoformat()

    prices = yf_adjclose(tickers, start=start, end=end)
    fwd_long = compute_forward_returns(prices, event_dates=event_dates, horizons=horizons)
    summary = summarize_forward_returns(fwd_long, benchmark=benchmark, horizons=horizons)

    # attach labels + small metadata
    for t in summary.get("tickers", []):
        t["label"] = universe.get(t["ticker"], t["ticker"])

    summary["event_dates"] = event_dates
    summary["horizons"] = list(horizons)
    summary["asof"] = datetime.now().isoformat(timespec="seconds")
    return summary

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

def build_why_2_0(trend_score, spx_above_50, spx_above_200, ndx_above_50, ndx_above_200, dji_above_50, dji_above_200,
                 scores, notes, vix_last, vix_pct, hy_spread_last, y10_last, atr_expansion,
                 rsp_pct, spy_pct, iwm_pct, hyg_pct, lqd_pct, uup_pct, gld_pct, tlt_pct,
                 btc_24, btc_7d, eth_24, eth_7d, crypto_agree):
    why = []

    # 1) One clean trend line
    why.append(
        f"Trend: score {trend_score}. "
        f"SPX {'Y' if spx_above_50 else 'N'}/50d, {'Y' if spx_above_200 else 'N'}/200d · "
        f"NDX {'Y' if ndx_above_50 else 'N'}/50d, {'Y' if ndx_above_200 else 'N'}/200d · "
        f"DJIA {'Y' if dji_above_50 else 'N'}/50d, {'Y' if dji_above_200 else 'N'}/200d."
    )

    # 2) Regime mechanics (relationship explanations)
    if not np.isnan(rsp_pct) and not np.isnan(spy_pct):
        if rsp_pct > spy_pct:
            why.append("Breadth: equal-weight outperforming → participation broadening (healthier risk-on).")
        else:
            why.append("Breadth: cap-weight leading → narrow leadership (more fragile risk-on).")

    if not np.isnan(iwm_pct) and not np.isnan(spy_pct):
        if iwm_pct > spy_pct:
            why.append("Leadership: small caps leading → liquidity/risk appetite improving.")
        else:
            why.append("Leadership: small caps lagging → liquidity cautious (risk-on less durable).")

    if not np.isnan(hyg_pct) and not np.isnan(lqd_pct):
        if hyg_pct > lqd_pct:
            why.append("Credit: high yield stronger than IG → funding conditions supportive.")
        else:
            why.append("Credit: high yield weaker than IG → funding stress can bleed into equities.")

    if not np.isnan(uup_pct) and not np.isnan(spy_pct):
        if uup_pct > 0 and spy_pct < 0:
            why.append("Liquidity: dollar up while stocks down → tightening backdrop (risk-off bias).")
        elif uup_pct < 0 and spy_pct > 0:
            why.append("Liquidity: dollar down while stocks up → easing tailwind (risk-on bias).")

    if not np.isnan(tlt_pct) and not np.isnan(spy_pct):
        if spy_pct < 0 and tlt_pct > 0:
            why.append("Positioning: bonds bid on equity weakness → defensive positioning active.")

    # 3) Stress snapshot (keep short)
    if not np.isnan(vix_last):
        why.append(f"Stress snapshot: VIX {fmt_num(vix_last,2)} (pctl {fmt_num(vix_pct,0)}), HY {fmt_num(hy_spread_last,2)}, 10Y {fmt_num(y10_last,2)}. "
                   f"{'ATR expanding' if atr_expansion else 'ATR stable'}.")

    # 4) Add best notes (already relationship-based)
    for n in notes[:4]:
        if n not in why:
            why.append(n)

    # 5) Crypto context (not scored)
    if not np.isnan(btc_24) and not np.isnan(eth_24):
        why.append(
            f"Crypto context (not scored): BTC {fmt_num(btc_24,2)}%/24h ({fmt_num(btc_7d,2)}%/7d), "
            f"ETH {fmt_num(eth_24,2)}%/24h ({fmt_num(eth_7d,2)}%/7d), "
            f"{'aligned' if crypto_agree else 'mixed'}."
        )

    return why[:10]

def build_interpretation(light, confidence, leverage_regime, chips, why):
    """
    5-tier interpretation:
    RED, ORANGE, YELLOW, GREEN, STAR
    """

    if light == "STAR":
        headline = "Clean risk-on (top-tier conditions)."
        what = [
            "Best backdrop for adding risk: trend + participation + low stress aligned.",
            "Leverage is allowed — still prefer pullbacks and clear leaders vs broad beta if you’ll be offline."
        ]
        watchouts = [
            "Complacency traps: a sudden VIX + credit stress pop can flip this fast.",
            "Don’t confuse a great regime with a guarantee — size like a pro, not like a tourist."
        ]

    elif light == "GREEN":
        headline = "Risk-on regime (trend beats stress)."
        what = [
            "Leverage is generally OK when trends hold and stress gauges stay contained.",
            "Prefer buying pullbacks over chasing; add on confirmation rather than emotion."
        ]
        watchouts = [
            "If VIX rises while HY spreads widen, downgrade fast (leverage gets punished).",
            "Sideways chop still bleeds leveraged ETFs even if the headline looks ‘fine’."
        ]

    elif light == "YELLOW":
        headline = "Mixed regime (selective risk)."
        what = [
            "Signals aren’t clean — reduce size, shorten holds, and demand confirmation.",
            "Focus on highest-quality setups; avoid broad leverage when tape is choppy."
        ]
        watchouts = [
            "Chop weeks are where leveraged ETFs bleed quietly.",
            "If stress worsens, assume downside tails are larger than they feel."
        ]

    elif light == "ORANGE":
        headline = "Elevated risk (defense favored)."
        what = [
            "Risk is skewed against leverage — prioritize capital preservation and flexibility.",
            "If you trade, make it smaller, faster, and more tactical (or hedge)."
        ]
        watchouts = [
            "Reflex rallies are common — don’t let them bait oversized risk-on leverage.",
            "If you’re wrong, you’ll know quickly; have exits defined up front."
        ]

    else:  # RED (and any unknown values)
        headline = "High-risk regime (stress beats trend)."
        what = [
            "Defense first: cash/hedges usually beat forcing upside here.",
            "Wait for stress to normalize and trend to repair before sizing up risk-on."
        ]
        watchouts = [
            "Bear market rallies are violent — don’t oversize inverses either.",
            "Avoid revenge leverage after big down days."
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
# Core: compute_stoplight (DROP-IN REPLACEMENT)
# ---------------------------

def compute_stoplight():
    # Pull data
    spx_s = get_fred_series_window("SP500", n=650)
    ndx_s = get_fred_series_window("NASDAQCOM", n=650)
    dji_s = get_fred_series_window("DJIA", n=650)

    vix_s = get_fred_series_window("VIXCLS", n=160)       # more room for percentile window
    y10_s = get_fred_series_window("DGS10", n=160)
    hy_s  = get_fred_series_window("BAMLH0A0HYM2", n=160)

    vix = series_stats(vix_s)
    y10 = series_stats(y10_s)
    hy  = series_stats(hy_s)

    # Index last values
    spx_last = float(spx_s.iloc[-1]) if len(spx_s) else np.nan
    ndx_last = float(ndx_s.iloc[-1]) if len(ndx_s) else np.nan
    dji_last = float(dji_s.iloc[-1]) if len(dji_s) else np.nan

    # Index momentum
    spx_5d  = pct_change_series(spx_s, 5)
    spx_21d = pct_change_series(spx_s, 21)
    ndx_5d  = pct_change_series(ndx_s, 5)
    ndx_21d = pct_change_series(ndx_s, 21)
    dji_5d  = pct_change_series(dji_s, 5)
    dji_21d = pct_change_series(dji_s, 21)

    # MAs
    spx_50  = sma_series(spx_s, 50)
    spx_200 = sma_series(spx_s, 200)
    ndx_50  = sma_series(ndx_s, 50)
    ndx_200 = sma_series(ndx_s, 200)
    dji_50  = sma_series(dji_s, 50)
    dji_200 = sma_series(dji_s, 200)

    # Trend score (now includes Dow)
    trend_score = 0

    spx_above_50  = (not np.isnan(spx_last) and not np.isnan(spx_50) and spx_last > spx_50)
    spx_above_200 = (not np.isnan(spx_last) and not np.isnan(spx_200) and spx_last > spx_200)
    ndx_above_50  = (not np.isnan(ndx_last) and not np.isnan(ndx_50) and ndx_last > ndx_50)
    ndx_above_200 = (not np.isnan(ndx_last) and not np.isnan(ndx_200) and ndx_last > ndx_200)
    dji_above_50  = (not np.isnan(dji_last) and not np.isnan(dji_50) and dji_last > dji_50)
    dji_above_200 = (not np.isnan(dji_last) and not np.isnan(dji_200) and dji_last > dji_200)

    for flag in [spx_above_50, spx_above_200, ndx_above_50, ndx_above_200, dji_above_50, dji_above_200]:
        trend_score += 1 if flag else -1

    # New additions
    vix_90 = get_fred_series_window("VIXCLS", n=90)
    vix_pct = percentile_rank(vix_90, vix["last"])

    spx_atr10 = atr_proxy_from_close(spx_s, 10)
    spx_atr30 = atr_proxy_from_close(spx_s, 30)
    atr_expansion = False
    if not np.isnan(spx_atr10) and not np.isnan(spx_atr30) and spx_atr30 != 0:
        atr_expansion = spx_atr10 > spx_atr30 * 1.15

    momentum_accel = (not np.isnan(ndx_5d) and not np.isnan(ndx_21d) and ndx_5d > ndx_21d)

    # Crypto (CONTEXT ONLY — NOT USED IN SCORING)
    btc_24 = crypto_return("BTC-USD", 24)
    eth_24 = crypto_return("ETH-USD", 24)
    btc_7d = crypto_return("BTC-USD", 24 * 7)
    eth_7d = crypto_return("ETH-USD", 24 * 7)
    crypto_agree = (not np.isnan(btc_24) and not np.isnan(eth_24) and (btc_24 * eth_24) > 0)

    # ---------------------------
    # LIVE TAPE + CROSS-ASSET (Yahoo)
    # ---------------------------
    xa = cross_asset_snapshot()
    pct = xa.get("pct", {})
    
    tape = tape_signal_from_pct(pct)
    tape_avg = tape.get("avg", np.nan)

    spy_pct = pct.get("SPY", np.nan)
    qqq_pct = pct.get("QQQ", np.nan)
    dia_pct = pct.get("DIA", np.nan)
    rsp_pct = pct.get("RSP", np.nan)
    iwm_pct = pct.get("IWM", np.nan)
    hyg_pct = pct.get("HYG", np.nan)
    lqd_pct = pct.get("LQD", np.nan)
    uup_pct = pct.get("UUP", np.nan)
    gld_pct = pct.get("GLD", np.nan)
    tlt_pct = pct.get("TLT", np.nan)
    uso_pct = pct.get("USO", np.nan)

    # ---------------------------
    # Regime 2.0 scoring
    # ---------------------------
    scores, notes = regime_2_0_scores(
        spx_above_50, spx_above_200,
        ndx_above_50, ndx_above_200,
        dji_above_50, dji_above_200,
        vix["last"], vix_pct, hy["last"], hy["d5"], y10["last"], atr_expansion,
        spy_pct, qqq_pct, dia_pct,
        rsp_pct, iwm_pct,
        hyg_pct, lqd_pct,
        uup_pct, gld_pct, tlt_pct, uso_pct,
        tape_avg,
    )

    net = regime_2_0_net(scores)

    # 5-tier output
    light, regime, icon = stoplight_5_tier(net)

    # Optional: quick “confidence” heuristic (you can refine later)
    # HIGH when signal is strong; MEDIUM when modest; LOW when close to neutral
    if abs(net) >= 5:
        confidence = "HIGH"
    elif abs(net) >= 3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Leverage regime (simple + consistent with your new model)
    leverage_regime = "FAVORABLE" if regime in ("LOW RISK", "NIRVANA") else ("UNFAVORABLE" if regime in ("HIGH RISK", "ELEVATED RISK") else "NEUTRAL")

    # Lever menu
    if regime in ("NIRVANA", "LOW RISK"):
        chips = ["TQQQ", "NVDL", "AMDL", "CONL", "Single-name > broad if you’ll be offline"]
    elif regime == "NEUTRAL":
        chips = ["Smaller size", "Shorter holds", "Prefer single-name leverage", "Wait for confirmation"]
    else:
        chips = ["PSQ / SQQQ", "HIBS (aggressive)", "Reduce/avoid leverage", "Cash is a position"]

    # Why (new builder)
    why = build_why_2_0(
        trend_score,
        spx_above_50, spx_above_200, ndx_above_50, ndx_above_200, dji_above_50, dji_above_200,
        scores, notes,
        vix["last"], vix_pct, hy["last"], y10["last"], atr_expansion,
        rsp_pct, spy_pct, iwm_pct, hyg_pct, lqd_pct, uup_pct, gld_pct, tlt_pct,
        btc_24, btc_7d, eth_24, eth_7d, crypto_agree
    )

    # ---------------------------
    # Metrics (dashboard)
    # ---------------------------
    metrics = [
        {
            "name": "Tape (live)",
            "value": f"Avg {fmt_num(tape_avg,2)}" + ("" if np.isnan(tape_avg) else "%"),
            "delta": (
                f"SPY {fmt_num((tape or {}).get('spy', np.nan), 2)} | "
                f"QQQ {fmt_num((tape or {}).get('qqq', np.nan), 2)} | "
                f"DIA {fmt_num((tape or {}).get('dia', np.nan), 2)}"
            ),
        },
        {
            "name": "Regime",
            "value": f"{icon} {regime}",
            "delta": f"Stoplight: {light} | Leverage: {leverage_regime}",
        },
        {
            "name": "Net",
            "value": str(int(net)),
            "delta": f"Trend pillar: {scores.get('trend',0)} | Stress: {scores.get('stress',0)} | Tape: {scores.get('tape',0)}",
        },
        {
            "name": "VIX",
            "value": fmt_num(vix["last"], 2),
            "delta": f"5D {fmt_delta(vix['d5'], 2)} | Pctl90D {fmt_num(vix_pct, 0)}",
        },
        {
            "name": "HY Spread",
            "value": fmt_num(hy["last"], 2),
            "delta": f"5D {fmt_delta(hy['d5'], 2)}",
        },
        {
            "name": "10Y",
            "value": fmt_num(y10["last"], 2),
            "delta": f"5D {fmt_delta(y10['d5'], 2)}",
        },
        {
            "name": "SPX",
            "value": fmt_num(spx_last, 2),
            "delta": f"5D {fmt_num(spx_5d, 2)}% | 21D {fmt_num(spx_21d, 2)}%",
        },
        {
            "name": "NDX",
            "value": fmt_num(ndx_last, 2),
            "delta": f"5D {fmt_num(ndx_5d, 2)}% | 21D {fmt_num(ndx_21d, 2)}%",
        },
        {
            "name": "DJIA",
            "value": fmt_num(dji_last, 2),
            "delta": f"5D {fmt_num(dji_5d, 2)}% | 21D {fmt_num(dji_21d, 2)}%",
        },
        {
            "name": "Vol regime",
            "value": "ATR expanding" if atr_expansion else "ATR stable",
            "delta": f"NDX accel: {momentum_accel}",
        },
        {
            "name": "Crypto (context)",
            "value": f"BTC {fmt_num(btc_24,2)}% / ETH {fmt_num(eth_24,2)}% (24h)",
            "delta": "Not in scoring",
        },
        {
            "name": "Breadth",
            "value": (
                f"RSP {fmt_num(rsp_pct,2)}" + ("" if np.isnan(rsp_pct) else "%") +
                f" vs SPY {fmt_num(spy_pct,2)}" + ("" if np.isnan(spy_pct) else "%")
            ),
            "delta": "Equal-weight vs cap-weight",
        },
        {
            "name": "Credit",
            "value": (
                f"HYG {fmt_num(hyg_pct,2)}" + ("" if np.isnan(hyg_pct) else "%") +
                f" vs LQD {fmt_num(lqd_pct,2)}" + ("" if np.isnan(lqd_pct) else "%")
            ),
            "delta": "High yield vs IG",
        },
        {
            "name": "USD / Gold / Bonds",
            "value": (
                f"UUP {fmt_num(uup_pct,2)}" + ("" if np.isnan(uup_pct) else "%") + " · " +
                f"GLD {fmt_num(gld_pct,2)}" + ("" if np.isnan(gld_pct) else "%") + " · " +
                f"TLT {fmt_num(tlt_pct,2)}" + ("" if np.isnan(tlt_pct) else "%")
            ),
            "delta": "Dollar / risk-hedges",
        },
    ]

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "light": light,
        "regime": regime,
        "confidence": confidence,
        "net": int(net),
        "trend_score": int(trend_score),
        "scores": scores,
        "leverage_regime": leverage_regime,
        "metrics": metrics,
        "why": why[:10],
        "chips": chips,
    }
# ---------------------------
# Routes
# ---------------------------

@app.get("/status")
def status():
    return compute_stoplight()


@app.get("/", response_class=HTMLResponse)
def homepage():
    try:
        data = compute_stoplight()
    except Exception as e:
        data = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "light": "YELLOW",
            "regime": "NEUTRAL",
            "confidence": "LOW",
            "net": "NA",
            "metrics": [],
            "chips": [],
            "why": [f"compute_stoplight error: {type(e).__name__}: {e}"],
        }

    light = data.get("light", "YELLOW")

    # 5-tier color map
    color_map = {
        "RED": "#ff4d4f",
        "ORANGE": "#ff9f43",
        "YELLOW": "#ffd166",
        "GREEN": "#2ee59d",
        "STAR": "#6ecbff",
    }
    color = color_map.get(light, "#ffd166")

    cards_html = "".join([
        f"""
        <div class="card">
          <div class="k">{m.get("name","")}</div>
          <div class="row">
            <div class="val">{m.get("value","")}</div>
            <div class="delta">{m.get("delta","")}</div>
          </div>
        </div>
        """ for m in data.get("metrics", [])
    ])

    why_html = "".join([f"<li>{x}</li>" for x in data.get("why", [])])
    menu_html = "".join([f'<span class="chip">{x}</span>' for x in data.get("chips", [])])

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
    .stamp {{ font-family:var(--mono); color:var(--muted2); font-size:12px; }}
    .grid {{ display:grid; grid-template-columns: 280px 1fr; gap:14px; margin-top:14px; }}
    @media(max-width:760px){{ .grid{{grid-template-columns:1fr;}} }}
    .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:18px; padding:16px; }}
    .light {{ width:210px; height:210px; border-radius:50%; margin:6px auto 10px; background:var(--accent); box-shadow:0 0 0 8px rgba(255,255,255,.06); }}
    .label {{ text-align:center; font-size:28px; font-weight:900; letter-spacing:1px; }}
    .sub {{ text-align:center; margin-top:6px; color:var(--muted); }}
    .k {{ color:var(--muted2); font-size:12px; letter-spacing:.10em; text-transform:uppercase; margin-bottom:8px; }}
    .cards {{ display:grid; grid-template-columns: 1fr 1fr; gap:10px; }}
    @media(max-width:760px){{ .cards{{grid-template-columns:1fr;}} }}
    .card {{ border:1px solid var(--border); border-radius:14px; padding:12px; background:rgba(0,0,0,.14); }}
    .row {{ display:flex; justify-content:space-between; gap:10px; align-items:baseline; }}
    .val {{ font-size:18px; font-weight:800; }}
    .delta {{ font-family:var(--mono); font-size:12px; color:var(--muted); text-align:right; }}
    ul {{ margin:6px 0 0 18px; color:var(--muted); line-height:1.45; }}
    .chip {{ display:inline-block; margin:6px 6px 0 0; padding:7px 10px; border-radius:999px; border:1px solid var(--border); background:rgba(0,0,0,.12); font-size:13px; }}
    .headline {{ font-size:16px; font-weight:800; margin-bottom:8px; }}
    .note {{ margin-top:12px; color:var(--muted2); font-size:12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="title">Market Stoplight</div>
      <div class="stamp">Updated: {data.get("timestamp","")}</div>
    </div>

    <div class="grid">
      <div class="panel">
        <div class="light"></div>
        <div class="label">{data.get("regime","NEUTRAL")}</div>
        <div class="sub">Stoplight: {data.get("light","YELLOW")} · Confidence: {data.get("confidence","NA")} · Net: {data.get("net","NA")}</div>

        <div class="note">
          <div class="k">Interpretation (Why)</div>
          <ul>
            {why_html}
          </ul>
        </div>
      </div>

      <div class="panel">
        <div class="headline">Dashboard</div>
        <div class="cards">
          {cards_html}
        </div>

        <div style="margin-top:14px;">
          <div class="k">Levers (menu, not advice)</div>
          {menu_html}
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""
