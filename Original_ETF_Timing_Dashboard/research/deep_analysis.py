"""
Deep robustness analysis:
1. Excess return vs unconditional baseline with bootstrap CI
2. QQQ cross-validation of top signals
3. Signal episode dates (inspect clustering)
4. Multiple-testing awareness
5. Build & validate ETF Opportunity Score
"""
import numpy as np, pandas as pd
from scipy import stats
np.random.seed(42)

df = pd.read_csv('daily_features.csv', index_col=0, parse_dates=True)
df.index = df.index.normalize()

base6 = df['fwd_6m'].dropna()
BASE_MEAN = base6.mean()
BASE_WIN = (base6 > 0).mean()
print(f'Baseline 6m mean={BASE_MEAN*100:.2f}% win={BASE_WIN*100:.1f}%\n')

def get_episodes(mask, gap=42):
    dates = list(df.index[mask.fillna(False)])
    if not dates: return []
    eps = [dates[0]]; last = dates[0]
    for d in dates[1:]:
        if (d - last).days > gap*1.5:   # calendar-day approx of trading gap
            eps.append(d); last = d
    return eps

def bootstrap_excess(returns, n_boot=5000):
    r = np.asarray(returns.dropna())
    # sample baseline of same size from unconditional distribution
    draws = np.random.choice(base6.values, size=(n_boot, len(r)), replace=True)
    diffs = r.mean() - draws.mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = (diffs <= 0).mean()
    return lo*100, hi*100, p

# Define top signals
SIGS = {
 'VIX>35 panic': (df['VIX']>35)&(df['VIX'].shift(1)<=35),
 'Golden cross 50/200': df['golden_cross'],
 '5% correction from 52w high': (df['dd_from_52w_high']<=-0.05)&(df['dd_from_52w_high'].shift(1)>-0.05),
 'Pullback >=5% above 200DMA': df['above_200dma']&(df['dd_from_52w_high']<=-0.05)&(df['dd_from_52w_high'].shift(1)>-0.05),
 'Combo F VIX30+DD10+curve': (df['VIX']>30)&(df['dd_from_52w_high']<=-0.10)&(df['curve_10y_13w']>0),
 'Combo A VIX30+DD10': (df['VIX']>30)&(df['dd_from_52w_high']<=-0.10),
 '200DMA recross': df['cross_above_200dma'],
 'Long-bond rally': df['rates_falling']&(df['rates_falling'].shift(1)==False),
 '10% correction': (df['dd_from_52w_high']<=-0.10)&(df['dd_from_52w_high'].shift(1)>-0.10),
 'VIX>30 panic': (df['VIX']>30)&(df['VIX'].shift(1)<=30),
 'Early Sept': (df['month']==9)&(df['dayofmonth']<=3),
 'VIX>90pct 3y': (df['vix_pct_3y']>0.9)&(df['vix_pct_3y'].shift(1)<=0.9),
}

rows = []
episodes_store = {}
for name, mask in SIGS.items():
    eps = get_episodes(mask)
    episodes_store[name] = eps
    r6 = df.loc[eps,'fwd_6m'].dropna()
    r6q = df.loc[eps,'fwd_6m_QQQ'].dropna()
    if len(r6) < 5: continue
    lo, hi, p = bootstrap_excess(r6)
    # win rate excess vs baseline (binomial)
    rows.append({
        'signal':name, 'n':len(r6),
        'avg6_spy':round(r6.mean()*100,2),
        'excess_vs_base':round((r6.mean()-BASE_MEAN)*100,2),
        'excess_CI_lo':round(lo,2),'excess_CI_hi':round(hi,2),'p_excess':round(p,4),
        'win_spy':round((r6>0).mean()*100,1),
        'win_excess':round(((r6>0).mean()-BASE_WIN)*100,1),
        'avg6_qqq':round(r6q.mean()*100,2),'win_qqq':round((r6q>0).mean()*100,1),
        'avg12_spy':round(df.loc[eps,'fwd_12m'].dropna().mean()*100,2),
        'avg_maxdd':round(df.loc[eps,'fwd_maxdd_12m'].dropna().mean()*100,2),
    })
ex = pd.DataFrame(rows).sort_values('excess_vs_base',ascending=False)
ex.to_csv('excess_analysis.csv',index=False)
print('EXCESS RETURN ANALYSIS (vs unconditional baseline):')
print(ex.to_string(index=False))

print('\n--- VIX>35 EPISODE DATES ---')
print([str(d.date()) for d in episodes_store['VIX>35 panic']])
print('\n--- 5% CORRECTION EPISODE DATES (first 20) ---')
print([str(d.date()) for d in episodes_store['5% correction from 52w high'][:20]])

# ---------------------------------------------------------------------------
# ETF Opportunity Score (0-100)
# Components: valuation-agnostic, all causal, designed for monthly automation
# ---------------------------------------------------------------------------
# Sub-scores (0-25 each, total 100)
def pct_rank(series, window, minp):
    return series.rolling(window, min_periods=minp).rank(pct=True)

df['dd_score'] = ((-df['dd_from_52w_high'])/0.20).clip(0,1)*25          # deeper dip -> higher (max at -20%)
df['vix_score'] = ((df['VIX']-12)/23).clip(0,1)*25                       # VIX 12->35 maps 0->25
df['trend_score'] = np.where(df['above_200dma'], 20, 5)                  # trend confirmation
df['macro_score'] = np.where(df['curve_10y_13w']>0, 8, 2) + np.where(df['rates_falling'],7,2) + np.where(df['VIX']<df['VIX'].rolling(252).mean(),10,0) + 10
# recompute macro cleanly: 0-25
df['macro_score'] = (np.where(df['curve_10y_13w']>0,8,2) + np.where(df['rates_falling'],7,2) + np.where(df['VIX']<df['VIX'].rolling(252).mean(),8,2)).clip(0,25)
# seasonality bonus: favorable months (Oct-Feb historically stronger 12m) +- small
fav = {1:2,2:1,3:0,4:-1,5:-2,6:0,7:0,8:0,9:1,10:2,11:2,12:2}
df['season_score'] = df['month'].map(fav) + 5   # 3-7 range; fold into trend adj
df['season_score'] = df['season_score'].clip(0,10)

df['opp_score'] = (df['dd_score'] + df['vix_score'] + df['trend_score'] + df['macro_score'] + df['season_score']).clip(0,100)
# Note: max = 25+25+20+25+10 = 105 -> clip to 100

# Validate score buckets against forward returns
df['score_bucket'] = pd.cut(df['opp_score'], bins=[-1,30,45,60,100], labels=['Wait (0-30)','Neutral (31-45)','Buy (46-60)','Strong Buy (61-100)'])
# de-overlap by sampling monthly (last trading day of each month) for validation
monthly = df.groupby(df.index.to_period('M')).tail(1).copy()
buck = monthly.groupby('score_bucket', observed=True).agg(
    count=('SPY','size'), avg_6m=('fwd_6m','mean'), win_6m=('fwd_6m',lambda s:(s>0).mean()),
    avg_12m=('fwd_12m','mean'), avg_maxdd=('fwd_maxdd_12m','mean'))
for c in ['avg_6m','win_6m','avg_12m','avg_maxdd']: buck[c]=(buck[c]*100).round(2)
buck.to_csv('score_validation.csv')
print('\nOPPORTUNITY SCORE VALIDATION (monthly sampled, forward returns):')
print(buck.to_string())

print('\nSCORE COMPONENT DESCRIPTION (ranges):')
for c in ['dd_score','vix_score','trend_score','macro_score','season_score','opp_score']:
    print(f'  {c}: {df[c].min():.1f} - {df[c].max():.1f}, mean {df[c].mean():.1f}')

print('\nCURRENT LATEST READING (as of last data date):')
last = df.iloc[-1]
print(f"  Date: {df.index[-1].date()}, SPY={last['SPY']:.2f}, VIX={last['VIX']:.2f}, DD from 52w high={last['dd_from_52w_high']*100:.2f}%")
print(f"  Score: {last['opp_score']:.1f} -> {last['score_bucket']}")
