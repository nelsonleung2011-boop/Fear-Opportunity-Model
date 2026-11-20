"""
Rebuild ETF Opportunity Score using empirically-calibrated non-linear components.
Validation uses NON-OVERLAPPING 6-month forward windows (sample every 126 trading days)
to avoid the overlap bias that distorted the first design.
"""
import numpy as np, pandas as pd

df = pd.read_csv('daily_features.csv', index_col=0, parse_dates=True)
df.index = df.index.normalize()

# ---------- Component calibration based on observed forward-return patterns ----------
# TREND (0-25): trend is the single most important regime gate
df['c_trend'] = np.where(df['above_200dma'], 22, 3)
df['c_trend'] += np.where(df['mom_12m']>0.10, 3, np.where(df['mom_12m']>0, 1, 0))
df['c_trend'] = df['c_trend'].clip(0,25)

# DIP (0-25): inverted-U - shallow dips best; deep dips only good WITH panic (handled by VIX)
dd = df['dd_from_52w_high']
df['c_dip'] = np.select(
    [dd>=-0.02, dd>=-0.05, dd>=-0.10, dd>=-0.15, dd< -0.15],
    [8, 22, 20, 10, 4], default=8)

# VIX (0-25): panic (VIX>=28) best; "creeping fear" 20-25 worst; low VIX neutral-good
v = df['VIX']
df['c_vix'] = np.select(
    [v<14, v<18, v<21, v<25, v<28, v<35, v>=35],
    [14, 16, 12, 5, 18, 24, 25], default=10)

# MACRO (0-15): falling rates + positive curve help modestly
df['c_macro'] = np.where(df['rates_falling'],8,3) + np.where(df['curve_10y_13w']>0,7,3)

# SEASONALITY (0-10): empirically stronger entry months for 6-12m horizon
seas = {1:9,2:8,3:6,4:4,5:3,6:6,7:6,8:5,9:9,10:9,11:9,12:9}
df['c_season'] = df['month'].map(seas)

df['opp_score'] = (df['c_trend']+df['c_dip']+df['c_vix']+df['c_macro']+df['c_season']).clip(0,100)

# ---------- Validate on NON-OVERLAPPING windows ----------
# take every 126th trading day (non-overlapping 6m blocks)
val = df.iloc[::126].dropna(subset=['fwd_6m']).copy()
val['bucket'] = pd.cut(val['opp_score'], bins=[-1,48,60,74,101],
                       labels=['Wait 0-48','Neutral 49-60','Buy 61-74','Strong Buy 75-100'])
g = val.groupby('bucket', observed=True).agg(n=('SPY','size'),avg6=('fwd_6m','mean'),
     win6=('fwd_6m',lambda s:(s>0).mean()),avg12=('fwd_12m','mean'),
     maxdd=('fwd_maxdd_12m','mean'))
for c in ['avg6','win6','avg12','maxdd']: g[c]=(g[c]*100).round(2)
print('NON-OVERLAPPING (every 126 trading days) VALIDATION:')
print(g.to_string())
print(f"\nOverall val baseline: avg6={val['fwd_6m'].mean()*100:.2f}% win={(val['fwd_6m']>0).mean()*100:.1f}% n={len(val)}")

# Also: monthly-sampled validation for comparison
m = df.groupby(df.index.to_period('M')).tail(1).copy()
m['bucket']=pd.cut(m['opp_score'],bins=[-1,48,60,74,101],labels=['Wait','Neutral','Buy','Strong Buy'])
g2=m.groupby('bucket',observed=True).agg(n=('SPY','size'),avg6=('fwd_6m','mean'),win6=('fwd_6m',lambda s:(s>0).mean()))
for c in ['avg6','win6']: g2[c]=(g2[c]*100).round(2)
print('\nMONTHLY-SAMPLED VALIDATION:')
print(g2.to_string())

# Component ranges and current reading
print('\nCOMPONENT RANGES:')
for c in ['c_trend','c_dip','c_vix','c_macro','c_season','opp_score']:
    print(f'  {c}: {df[c].min():.0f}-{df[c].max():.0f}, mean {df[c].mean():.1f}')

last=df.iloc[-1]
print(f"\nLATEST {df.index[-1].date()}: SPY={last['SPY']:.1f} VIX={last['VIX']:.1f} DD={last['dd_from_52w_high']*100:.1f}% trend={'above' if last['above_200dma'] else 'below'} 200DMA")
print(f"  Components: trend={last['c_trend']:.0f} dip={last['c_dip']:.0f} vix={last['c_vix']:.0f} macro={last['c_macro']:.0f} season={last['c_season']:.0f}")
print(f"  SCORE={last['opp_score']:.1f}")

# Save score history for chart and thresholds
df[['opp_score','c_trend','c_dip','c_vix','c_macro','c_season']].to_csv('score_history.csv')
val[['opp_score','fwd_6m','bucket']].to_csv('score_validation_clean.csv')
