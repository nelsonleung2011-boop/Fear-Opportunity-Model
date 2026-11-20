"""
Fetch historical market data via Yahoo Finance chart API (no auth required).
Data: SPY, QQQ, VOO, SCHG, VUG, IWY prices; ^VIX; ^GSPC; macro proxies.
Adjusted close used for total-return-like series.
"""
import urllib.request, ssl, json, time, os
import pandas as pd
import numpy as np

ctx = ssl.create_default_context()
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# period1 = 1999-01-01, period2 = 2026-11-16
P1 = int(pd.Timestamp('1999-01-01').timestamp())
P2 = int(pd.Timestamp('2026-11-16').timestamp())

SYMBOLS = ['SPY','QQQ','VOO','SCHG','VUG','IWY','^VIX','^GSPC','^VIX9D','TLT','IEF','SHY','DGS10','DGS2']

def fetch(symbol, retries=3):
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?period1={P1}&period2={P2}&interval=1d&events=div%2Csplit'
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=40, context=ctx) as r:
                data = json.load(r)
            res = data['chart']['result'][0]
            ts = res['timestamp']
            q = res['indicators']['quote'][0]
            adj = res['indicators'].get('adjclose', [{}])[0].get('adjclose', q['close'])
            df = pd.DataFrame({
                'date': pd.to_datetime(ts, unit='s', utc=True).tz_localize(None),
                'open': q['open'], 'high': q['high'], 'low': q['low'],
                'close': q['close'], 'adj_close': adj, 'volume': q['volume'],
            })
            df = df.dropna(subset=['close']).set_index('date')
            # add dividends/splits info
            ev = res.get('events', {})
            print(f'  {symbol}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}')
            return df
        except Exception as e:
            print(f'  {symbol} attempt {i+1} failed: {type(e).__name__}: {e}')
            time.sleep(2)
    return None

import urllib.parse
os.makedirs('data', exist_ok=True)
all_data = {}
for s in SYMBOLS:
    df = fetch(s)
    if df is not None:
        all_data[s] = df
        df.to_csv(f'data/{s.replace("^","_").lower()}.csv')
    time.sleep(0.5)

# FRED macro data via fredgraph CSV (VIX, 10Y-2Y, Fed funds, CPI, ISM)
FRED_SERIES = {
    'VIXCLS': 'vix_fred',
    'T10Y2Y': 't10y2y',
    'DFF': 'fedfunds',
    'FEDFUNDS': 'fedfunds_monthly',
    'CPIAUCSL': 'cpi',
    'MANEMP': 'ism_proxy',       # manufacturing employment (monthly, ISM proxy)
    'UMCSENT': 'umich_sentiment',
}
def fetch_fred(series_id, name):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=40, context=ctx) as r:
            txt = r.read().decode('utf-8', errors='ignore')
        from io import StringIO
        df = pd.read_csv(StringIO(txt))
        df.columns = ['date','value']
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna().set_index('date')
        df.to_csv(f'data/fred_{name}.csv')
        print(f'  FRED {series_id}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}')
        return df
    except Exception as e:
        print(f'  FRED {series_id} failed: {type(e).__name__}: {e}')
        return None

print('\nFetching FRED macro series...')
for sid, name in FRED_SERIES.items():
    fetch_fred(sid, name)
    time.sleep(0.5)

print('\nDone. Files in data/:')
for f in sorted(os.listdir('data')):
    print(' ', f)
