from fastapi import FastAPI
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = FastAPI()

FRED_KEY = os.getenv("FRED_API_KEY")

def get_fred_series(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_KEY,
        "file_type": "json"
    }
    r = requests.get(url, params=params)
    data = r.json()["observations"]
    df = pd.DataFrame(data)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df["value"].dropna().iloc[-1]

def get_crypto_return(product="BTC-USD", hours=24):
    granularity = 3600
    limit = hours + 1
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    params = {"granularity": granularity}
    r = requests.get(url, params=params)
    data = pd.DataFrame(r.json(), columns=["time","low","high","open","close","volume"])
    data = data.sort_values("time")
    if len(data) < limit:
        return 0
    old = data.iloc[-limit]["close"]
    new = data.iloc[-1]["close"]
    return (new - old) / old * 100

def compute_stoplight():
    btc_24 = get_crypto_return("BTC-USD", 24)
    eth_24 = get_crypto_return("ETH-USD", 24)

    vix = get_fred_series("VIXCLS")
    yield10 = get_fred_series("DGS10")
    hy = get_fred_series("BAMLH0A0HYM2")

    score = 0

    # Crypto impulse
    if btc_24 >= 5:
        score += 2
    elif btc_24 >= 2:
        score += 1
    elif btc_24 <= -5:
        score -= 2
    elif btc_24 <= -2:
        score -= 1

    if eth_24 * btc_24 > 0:
        score += 1
    else:
        score -= 1

    # Stress levers
    if vix >= 30:
        score -= 3
    elif vix >= 25:
        score -= 1

    if hy > 6:
        score -= 2

    if yield10 > 4.5:
        score -= 1

    if score >= 2:
        light = "GREEN"
    elif score <= -2:
        light = "RED"
    else:
        light = "YELLOW"

    return {
        "timestamp": str(datetime.now()),
        "btc_24h": round(btc_24,2),
        "eth_24h": round(eth_24,2),
        "vix": vix,
        "10y": yield10,
        "hy_spread": hy,
        "score": score,
        "light": light
    }

@app.get("/status")
def status():
    return compute_stoplight()

@app.get("/")
def homepage():
    data = compute_stoplight()
    color = {"GREEN":"#0f0","YELLOW":"#ff0","RED":"#f00"}[data["light"]]
    return f"""
    <html>
    <body style="background:black;color:white;font-family:Arial;text-align:center;">
    <h1>Market Stoplight</h1>
    <div style="margin:auto;width:200px;height:200px;border-radius:100px;background:{color};"></div>
    <h2>{data["light"]}</h2>
    <p>Score: {data["score"]}</p>
    <p>BTC 24h: {data["btc_24h"]}%</p>
    <p>ETH 24h: {data["eth_24h"]}%</p>
    <p>VIX: {data["vix"]}</p>
    <p>10Y: {data["10y"]}</p>
    <p>HY Spread: {data["hy_spread"]}</p>
    </body>
    </html>
    """
