"""
Generate compact JSON payload for the standalone interactive dashboard.
Embeds all daily series + precomputed signal episodes so the HTML needs no external files.
"""
import json
import pandas as pd
import numpy as np

s = pd.read_csv('score_history.csv', index_col=0, parse_dates=True)
d = pd.read_csv('daily_features.csv', index_col=0, parse_dates=True)
assert (s.index == d.index).all()

def arr(series, nd=3):
    return [None if pd.isna(x) else round(float(x), nd) for x in series]

def barr(series):
    return [0 if pd.isna(x) or not bool(x) else 1 for x in series]

P = {}
P['dates'] = [x.strftime('%Y-%m-%d') for x in d.index]
P['n'] = len(d)

# --- score components (source of truth: score_history.csv) ---
for c in ['opp_score','c_trend','c_dip','c_vix','c_macro','c_season']:
    P[c] = arr(s[c], 2)

# --- market series ---
series_map = {
    'SPY':('SPY',2),'QQQ':('QQQ',2),'VIX':('VIX',2),'IRX':('IRX',3),
    'TNX':('TNX',3),'TLT':('TLT',2),'dd':('dd_from_52w_high',4),
    'curve':('curve_10y_13w',3),'mom12':('mom_12m',4),
    'vixpct':('vix_pct_3y',4),'fwd3':('fwd_3m',4),'fwd6':('fwd_6m',4),
    'fwd12':('fwd_12m',4),'maxdd12':('fwd_maxdd_12m',4),
}
for key,(col,nd) in series_map.items():
    P[key] = arr(d[col], nd)

# indicator reference series (used for the initial embedded preview)
P['high52'] = arr(d['roll_high_252'], 2)
P['sma200'] = arr(d['sma200'], 2)
P['sma50'] = arr(d['sma50'], 2)
P['tltma126'] = arr(d['TLT'].rolling(126, min_periods=126).mean(), 2)

P['month'] = [int(x) for x in d['month']]
P['year'] = [int(x) for x in d['year']]

# --- boolean state / trigger arrays ---
bools = {
 'above200': d['above_200dma'],
 'sma50gt200': d['sma50'] > d['sma200'],
 'golden': d['golden_cross'],
 'cross200': d['cross_above_200dma'],
 'ratesFalling': d['rates_falling'],
 'curveSteepen': d['curve_steepen'],
 'fedCut': d['fed_cut_proxy'],
 'momCross': d['mom_12m_cross_up'],
}
# level conditions
bools['vix25'] = d['VIX'] > 25
bools['vix30'] = d['VIX'] > 30
bools['vix35'] = d['VIX'] > 35
for thr in [5,10,15,20]:
    bools[f'dd{thr}'] = d['dd_from_52w_high'] <= -thr/100
# crossing conditions
bools['vix35cross'] = (d['VIX']>35)&(d['VIX'].shift(1)<=35)
bools['vix30cross'] = (d['VIX']>30)&(d['VIX'].shift(1)<=30)
bools['vix25cross'] = (d['VIX']>25)&(d['VIX'].shift(1)<=25)
for thr in [5,10,15,20]:
    bools[f'dd{thr}cross'] = (d['dd_from_52w_high']<=-thr/100)&(d['dd_from_52w_high'].shift(1)>-thr/100)
bools['pullback5above'] = d['above_200dma']&(d['dd_from_52w_high']<=-0.05)&(d['dd_from_52w_high'].shift(1)>-0.05)
bools['comboA'] = (d['VIX']>30)&(d['dd_from_52w_high']<=-0.10)
bools['comboF'] = bools['comboA']&(d['curve_10y_13w']>0)
bools['comboC'] = bools['vix25cross']|bools['dd5cross']
bools['comboD'] = (d['vix_pct_3y']>0.85)&(d['dd_from_52w_high']<=-0.08)
bools['fearTop10'] = d['vix_pct_3y']>0.9
bools['vixSpikeFade'] = (d['VIX'].rolling(10).max()>30)&(d['VIX']<d['VIX'].rolling(10).max()*0.8)&(d['VIX']<d['VIX'].shift(1))
for k,v in bools.items():
    P['b_'+k] = barr(v)

# --- manual-input reference values (latest date) ---
last = d.iloc[-1]
P['refs'] = {
 'high52': round(float(d['roll_high_252'].iloc[-1]),2),
 'sma200': round(float(last['sma200']),2),
 'price252': round(float(d['SPY'].iloc[-253]),2),
 'tltma126': round(float(d['TLT'].rolling(126).mean().iloc[-1]),2),
 'latestDate': d.index[-1].strftime('%Y-%m-%d'),
 'latestSPY': round(float(last['SPY']),2),
 'latestVIX': round(float(last['VIX']),2),
 'latest10Y': round(float(last['TNX']),2),
 'latest13W': round(float(last['IRX']),2),
 'latestTLT': round(float(last['TLT']),2),
}

# --- precomputed signal episodes for backtest explorer ---
def cluster(idx_list, gap):
    """Cluster by calendar-day gap (matches deep_analysis.py used for the report)."""
    if not idx_list: return []
    out=[idx_list[0]]; last_i=idx_list[0]
    for i in idx_list[1:]:
        if (d.index[i] - d.index[last_i]).days > gap:
            out.append(i); last_i=i
    return out

sig_defs = [
 ('VIX > 35 (panic entry)', bools['vix35cross'], 63, 'Volatility'),
 ('VIX > 30 (panic entry)', bools['vix30cross'], 63, 'Volatility'),
 ('VIX > 25 (panic entry)', bools['vix25cross'], 63, 'Volatility'),
 ('5% pullback from 52w high', bools['dd5cross'], 63, 'Drawdown'),
 ('10% pullback from 52w high', bools['dd10cross'], 63, 'Drawdown'),
 ('15% pullback from 52w high', bools['dd15cross'], 63, 'Drawdown'),
 ('20% pullback from 52w high', bools['dd20cross'], 63, 'Drawdown'),
 ('5% pullback while above 200DMA', bools['pullback5above'], 63, 'Trend'),
 ('Golden cross (50/200)', bools['golden'], 63, 'Trend'),
 ('Price recrosses above 200DMA', bools['cross200'], 63, 'Trend'),
 ('12m momentum turns positive', bools['momCross'], 63, 'Trend'),
 ('Combo A: VIX>30 + DD>=10%', (d['VIX']>30)&(bools['comboA']), 63, 'Combination'),
 ('Combo F: + curve positive', bools['comboF'], 63, 'Combination'),
 ('Combo C: VIX>25 or DD>=5%', bools['comboC'], 63, 'Combination'),
 ('VIX top 10% of 3y range', bools['fearTop10'], 63, 'Sentiment'),
 ('Long-bond rally (TLT>6m MA)', bools['ratesFalling']&(~bools['ratesFalling'].shift(1).fillna(False).astype(bool)), 63, 'Macro'),
 ('Buy early September', (d['month']==9)&(d['dayofmonth']<=3), 63, 'Seasonality'),
 ('Buy early December', (d['month']==12)&(d['dayofmonth']<=3), 63, 'Seasonality'),
]

base6 = d['fwd_6m'].mean()
episodes = []
for name, mask, gap, cat in sig_defs:
    idxs = [i for i,v in enumerate(mask.fillna(False)) if v]
    eps = cluster(idxs, gap)
    rows=[]
    for i in eps:
        rows.append({
            'i': i, 'date': P['dates'][i],
            'r3': P['fwd3'][i], 'r6': P['fwd6'][i], 'r12': P['fwd12'][i],
            'dd': P['maxdd12'][i]
        })
    r6s = [r['r6'] for r in rows if r['r6'] is not None]
    episodes.append({
        'name':name, 'cat':cat, 'gap':gap,
        'n':len(rows), 'rows':rows,
        'win6': round(100*np.mean([x>0 for x in r6s]),1) if r6s else None,
        'avg6': round(100*np.mean(r6s),2) if r6s else None,
        'avg3': round(100*np.mean([r['r3'] for r in rows if r['r3'] is not None]),2),
        'avg12': round(100*np.mean([r['r12'] for r in rows if r['r12'] is not None]),2),
        'excess6': round(100*(np.mean(r6s)-base6),2) if r6s else None,
    })
P['signals'] = episodes
P['base'] = {'avg6':round(100*base6,2),'win6':round(100*(d['fwd_6m']>0).mean(),1)}

with open('dashboard_data.json','w') as f:
    json.dump(P,f,separators=(',',':'))
import os
print('dashboard_data.json written:', round(os.path.getsize('dashboard_data.json')/1024,1),'KB')
print('dates:',P['n'],'signals:',len(episodes))
for e in episodes:
    print(f"  {e['name'][:38]:38s} n={e['n']:3d} win6={e['win6']} avg6={e['avg6']}")
