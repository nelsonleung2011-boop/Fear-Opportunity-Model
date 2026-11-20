"""
ETF Entry-Timing Backtest Engine
================================
Evaluates 6 strategy families for long-entry timing into broad-market US ETFs.
Primary instrument: SPY (1999-2026, 27 yrs). Secondary: QQQ.
Methodology: event-driven forward returns with overlap de-duplication,
             unconditional benchmark, sub-period stability, t-stats.
No look-ahead: all signals use data up to and including signal date t;
forward returns measured t+1 .. t+H (entry at close of t).
"""
import json, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.normalize()   # strip intraday timestamps -> date only
    return df[~df.index.duplicated(keep='last')].sort_index()

spy = load('data/spy.csv')['adj_close'].rename('SPY')
qqq = load('data/qqq.csv')['adj_close'].rename('QQQ')
vix = load('data/_vix.csv')['close'].rename('VIX')
irx = load('data/irx.csv')['close'].rename('IRX')   # 13-week T-bill (~Fed policy proxy)
tnx = load('data/tnx.csv')['close'].rename('TNX')   # 10-year yield
fvx = load('data/fvx.csv')['close'].rename('FVX')   # 5-year yield
tlt = load('data/tlt.csv')['adj_close'].rename('TLT')

# Master frame on SPY trading calendar, forward-filled for lower-frequency series
df = pd.DataFrame(index=spy.index)
df['SPY'] = spy
df['QQQ'] = qqq.reindex(df.index).ffill()
df['VIX'] = vix.reindex(df.index).ffill()
df['IRX'] = irx.reindex(df.index).ffill()
df['TNX'] = tnx.reindex(df.index).ffill()
df['FVX'] = fvx.reindex(df.index).ffill()
df['TLT'] = tlt.reindex(df.index).ffill()
df = df.dropna(subset=['SPY'])

START = '2000-01-01'
df = df[df.index >= START].copy()

# ---------------------------------------------------------------------------
# Feature construction (all causal / trailing)
# ---------------------------------------------------------------------------
px = df['SPY']
df['ret_1d'] = px.pct_change()
for h, label in [(21,'1m'),(63,'3m'),(126,'6m'),(252,'12m')]:
    df[f'fwd_{label}'] = px.shift(-h) / px - 1.0          # forward total return
    df[f'fwd_{label}_QQQ'] = df['QQQ'].shift(-h) / df['QQQ'] - 1.0

# Forward max drawdown over next 12m (worst peak-to-trough from entry)
future_min_252 = px.shift(-1).rolling(252, min_periods=200).min().shift(-251)
# simpler: min of prices in t+1..t+252 using reversed rolling
rev = px[::-1]
fwd_min = rev.rolling(252, min_periods=200).min()[::-1].shift(-1)
df['fwd_maxdd_12m'] = fwd_min / px - 1.0

# Trailing features
df['roll_high_252'] = px.rolling(252, min_periods=200).max()
df['dd_from_52w_high'] = px / df['roll_high_252'] - 1.0
df['sma50'] = px.rolling(50).mean()
df['sma200'] = px.rolling(200).mean()
df['sma10m'] = px.rolling(210).mean()
df['above_200dma'] = px > df['sma200']
df['golden_cross'] = (df['sma50'] > df['sma200']) & (df['sma50'].shift(1) <= df['sma200'].shift(1))
df['cross_above_200dma'] = (px > df['sma200']) & (px.shift(1) <= df['sma200'].shift(1))
df['mom_12m'] = px / px.shift(252) - 1.0
df['mom_12m_cross_up'] = (df['mom_12m'] > 0) & (df['mom_12m'].shift(1) <= 0)
df['vix_pct_3y'] = df['VIX'].rolling(756, min_periods=400).rank(pct=True)

# Macro proxies
df['curve_10y_13w'] = df['TNX'] - df['IRX']           # steepness proxy
df['curve_steepen'] = (df['curve_10y_13w'] > 0) & (df['curve_10y_13w'].shift(63) <= 0)
df['fed_cut_proxy'] = (df['IRX'].shift(21) - df['IRX'] > 0.25) & (df['IRX'] < df['IRX'].rolling(126).mean())
df['rates_falling'] = df['TLT'] > df['TLT'].rolling(126).mean()

# Calendar
df['year'] = df.index.year
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['dayofmonth'] = df.index.day
# Presidential cycle: 1=post-election yr,2=midterm,3=pre-election,4=election
df['pres_year'] = ((df['year'] - 1) % 4) + 1

# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------
H = 126  # primary horizon 6m
H12 = 252

def cluster_signals(signal_dates, gap=21):
    """Collapse signals within `gap` trading days into one episode (take first)."""
    if len(signal_dates) == 0:
        return []
    episodes = [signal_dates[0]]
    last = signal_dates[0]
    all_dates = list(df.index)
    for d in signal_dates[1:]:
        idx_d = all_dates.index(d) if d in all_dates else None
        idx_l = all_dates.index(last)
        if idx_d is not None and (idx_d - idx_l) > gap:
            episodes.append(d)
            last = d
    return episodes

def evaluate(signal_mask, name, category, gap=21, min_n=8):
    sig_dates = list(df.index[signal_mask.fillna(False)])
    episodes = cluster_signals(sig_dates, gap=gap)
    # use episode-entry returns
    sub = df.loc[episodes]
    r6 = sub['fwd_6m'].dropna()
    r12 = sub['fwd_12m'].dropna()
    r3 = sub['fwd_3m'].dropna()
    n = len(r6)
    if n < min_n:
        return None
    win6 = (r6 > 0).mean()
    win12 = (r12 > 0).mean() if len(r12) else np.nan
    # benchmark: unconditional over full sample
    bm6 = df['fwd_6m'].dropna()
    bm12 = df['fwd_12m'].dropna()
    # de-mean benchmark to same episode count via bootstrap of win rate
    t_r6 = stats.ttest_1samp(r6, 0).statistic if n > 1 else np.nan
    t_r12 = stats.ttest_1samp(r12, 0).statistic if len(r12) > 1 else np.nan
    # binom test win rate vs 50%
    p_win6 = stats.binomtest(int((r6>0).sum()), n, 0.5, alternative='greater').pvalue
    maxdd = sub['fwd_maxdd_12m'].dropna().mean()
    worst_dd = sub['fwd_maxdd_12m'].dropna().min()
    # annual occurrences
    yrs = (df.index.max() - df.index.min()).days / 365.25
    ann = n / yrs
    # sub-period stability (split at 2013)
    e_early = [d for d in episodes if d.year <= 2012]
    e_late = [d for d in episodes if d.year >= 2013]
    early_mean = df.loc[e_early,'fwd_6m'].mean() if len(e_early)>=3 else np.nan
    late_mean = df.loc[e_late,'fwd_6m'].mean() if len(e_late)>=3 else np.nan
    stable = (not np.isnan(early_mean)) and (not np.isnan(late_mean)) and (early_mean>0) and (late_mean>0)
    return {
        'name': name, 'category': category, 'n_signals_raw': len(sig_dates),
        'n_episodes': n, 'ann_occurrences': round(ann,2),
        'win_rate_6m': round(win6*100,1), 'win_rate_12m': round(win12*100,1),
        'avg_6m': round(r6.mean()*100,2), 'avg_12m': round(r12.mean()*100,2) if len(r12) else np.nan,
        'median_6m': round(r6.median()*100,2),
        'avg_3m': round(r3.mean()*100,2) if len(r3) else np.nan,
        't_stat_6m': round(t_r6,2), 't_stat_12m': round(t_r12,2),
        'p_win6_vs_50': round(p_win6,4),
        'avg_maxdd_12m': round(maxdd*100,2), 'worst_maxdd_12m': round(worst_dd*100,2),
        'early_mean_6m': round(early_mean*100,2) if not np.isnan(early_mean) else None,
        'late_mean_6m': round(late_mean*100,2) if not np.isnan(late_mean) else None,
        'stable': stable,
    }

results = []

# Baseline
base6 = df['fwd_6m'].dropna(); base12 = df['fwd_12m'].dropna()
baseline = {
    'name':'Unconditional benchmark (buy any day)','category':'Benchmark',
    'n_episodes':len(base6),'ann_occurrences':252.0,
    'win_rate_6m':round((base6>0).mean()*100,1),'win_rate_12m':round((base12>0).mean()*100,1),
    'avg_6m':round(base6.mean()*100,2),'avg_12m':round(base12.mean()*100,2),
    'avg_maxdd_12m':round(df['fwd_maxdd_12m'].dropna().mean()*100,2),
}
results.append(baseline)

# ---------------------------------------------------------------------------
# 1. Seasonality
# ---------------------------------------------------------------------------
# Best months to buy (first trading day of each month)
for m in range(1,13):
    mask = (df['month']==m) & (df['dayofmonth']<=3) & (df.index.to_series().diff().dt.days.notna())
    r = evaluate(mask, f'Buy early {pd.Timestamp(2020,m,1).month_name()}', 'Seasonality', gap=21)
    if r: results.append(r)

# Halloween / Sell-in-May: buy first trading day of November
nov_mask = (df['month']==11)&(df['dayofmonth']<=3)
r = evaluate(nov_mask, 'Halloween effect (buy early Nov, hold ~6m)', 'Seasonality', gap=200)
if r: results.append(r)
may_mask = (df['month']==5)&(df['dayofmonth']<=3)
r = evaluate(may_mask, 'Sell-in-May entry (buy early May)', 'Seasonality', gap=200)
if r: results.append(r)

# Presidential cycle (enter Jan of each cycle year, hold 12m)
for py,label in [(1,'Post-election yr 1'),(2,'Midterm yr 2'),(3,'Pre-election yr 3'),(4,'Election yr 4')]:
    mask = (df['pres_year']==py)&(df['month']==1)&(df['dayofmonth']<=5)
    # for 12m returns evaluate separately
    r = evaluate(mask, f'Presidential cycle: {label} (buy Jan)', 'Seasonality', gap=300)
    if r: results.append(r)

# Tax-loss selling rebound: buy last week of December / first week January
tax_mask = ((df['month']==12)&(df['dayofmonth']>=24)) | ((df['month']==1)&(df['dayofmonth']<=5))
r = evaluate(tax_mask, 'Tax-loss selling rebound (late Dec / early Jan)', 'Seasonality', gap=60)
if r: results.append(r)

# End-of-quarter weakness: last 3 trading days of Mar/Jun/Sep/Dec
df['is_qend'] = df.index.to_series().dt.month.isin([3,6,9,12])
# approximate: dayofmonth >= 27
qe_mask = df['is_qend'] & (df['dayofmonth']>=27)
r = evaluate(qe_mask, 'End-of-quarter weakness (late Mar/Jun/Sep/Dec)', 'Seasonality', gap=60)
if r: results.append(r)

# ---------------------------------------------------------------------------
# 2. Drawdown-based entries (first crossing of threshold from 52w high)
# ---------------------------------------------------------------------------
for thr,label in [(-0.05,'5%'),(-0.10,'10%'),(-0.15,'15%'),(-0.20,'20%')]:
    cond = (df['dd_from_52w_high'] <= thr) & (df['dd_from_52w_high'].shift(1) > thr)
    r = evaluate(cond, f'Buy after {label} correction from 52w high', 'Drawdown', gap=42)
    if r: results.append(r)
# Deeper: buy after 10% correction AND recovering 3% off lows (causal approx)
cond = (df['dd_from_52w_high']<=-0.10) & (df['ret_1d']>0) & (df['ret_1d'].rolling(5).sum()>0.02)
r = evaluate(cond, '10% correction + early stabilization (5d bounce)', 'Drawdown', gap=42)
if r: results.append(r)

# ---------------------------------------------------------------------------
# 3. VIX regime entries
# ---------------------------------------------------------------------------
for thr in [25,30,35]:
    cond = (df['VIX']>thr) & (df['VIX'].shift(1)<=thr)
    r = evaluate(cond, f'VIX crosses above {thr} (panic entry)', 'Volatility', gap=42)
    if r: results.append(r)
# VIX spike then decline: VIX peaked >30 in last 10d, now < peak*0.8 and falling
vix_peak10 = df['VIX'].rolling(10).max()
cond = (vix_peak10>30) & (df['VIX']<vix_peak10*0.8) & (df['VIX']<df['VIX'].shift(1)) & (df['VIX'].shift(1)>=df['VIX'].shift(2))
r = evaluate(cond, 'VIX spike >30 then 20% decline (panic subsiding)', 'Volatility', gap=42)
if r: results.append(r)
# VIX > 90th pct 3y (fear extreme)
cond = (df['vix_pct_3y']>0.9) & (df['vix_pct_3y'].shift(1)<=0.9)
r = evaluate(cond, 'VIX > 90th percentile (3y fear extreme)', 'Volatility', gap=42)
if r: results.append(r)

# ---------------------------------------------------------------------------
# 4. Trend-following entries
# ---------------------------------------------------------------------------
r = evaluate(df['cross_above_200dma'], 'Price recovers above 200DMA', 'Trend', gap=42)
if r: results.append(r)
r = evaluate(df['golden_cross'], 'Golden cross (50DMA crosses above 200DMA)', 'Trend', gap=63)
if r: results.append(r)
cond = (px>df['sma10m']) & (px.shift(1)<=df['sma10m'].shift(1))
r = evaluate(cond, 'Price recovers above 10-month MA', 'Trend', gap=42)
if r: results.append(r)
r = evaluate(df['mom_12m_cross_up'], '12-month momentum turns positive', 'Trend', gap=63)
if r: results.append(r)
# Pullback in uptrend: above 200dma but 5% off recent high
cond = df['above_200dma'] & (df['dd_from_52w_high']<=-0.05) & (df['dd_from_52w_high'].shift(1)>-0.05)
r = evaluate(cond, 'Pullback >=5% while above 200DMA', 'Trend', gap=42)
if r: results.append(r)

# ---------------------------------------------------------------------------
# 5. Macro regime filters
# ---------------------------------------------------------------------------
r = evaluate(df['curve_steepen'], 'Yield curve steepens from inverted (10Y-13W)', 'Macro', gap=63)
if r: results.append(r)
r = evaluate(df['fed_cut_proxy'], 'Fed-cutting proxy: 13W bill yield falling sharply', 'Macro', gap=42)
if r: results.append(r)
cond = df['rates_falling'] & (df['rates_falling'].shift(1)==False)
r = evaluate(cond, 'Long-bond rally begins (TLT > 6m MA, disinflation proxy)', 'Macro', gap=63)
if r: results.append(r)
# ISM proxy unavailable - document

# ---------------------------------------------------------------------------
# 6. Sentiment extremes (VIX percentile as fear proxy - documented limitation)
# ---------------------------------------------------------------------------
r = evaluate(df['vix_pct_3y']>0.85, 'Sentiment: VIX in top 15% of 3y range (fear)', 'Sentiment', gap=42)
if r: results.append(r)

# ---------------------------------------------------------------------------
# 7. COMBINATION SIGNALS (Tier B)
# ---------------------------------------------------------------------------
comb_defs = {
 'Combo A: VIX>30 AND DD<=-10%': (df['VIX']>30)&(df['dd_from_52w_high']<=-0.10),
 'Combo B: VIX>30 AND DD<=-10% AND >200DMA': (df['VIX']>30)&(df['dd_from_52w_high']<=-0.10)&df['above_200dma'],
 'Combo C: VIX>25 AND DD<=-5% (first cross of either)': ((df['VIX']>25)&(df['VIX'].shift(1)<=25))|((df['dd_from_52w_high']<=-0.05)&(df['dd_from_52w_high'].shift(1)>-0.05)),
 'Combo D: Fear (VIX>85pct) AND DD<=-8%': (df['vix_pct_3y']>0.85)&(df['dd_from_52w_high']<=-0.08),
 'Combo E: DD<=-10% AND above 200DMA AND rates falling': (df['dd_from_52w_high']<=-0.10)&df['above_200dma']&df['rates_falling'],
 'Combo F: VIX>30 AND DD<=-10% AND curve steepening': (df['VIX']>30)&(df['dd_from_52w_high']<=-0.10)&(df['curve_10y_13w']>0),
 'Combo G: Post-200DMA-cross + 10% dip within 60d': df['cross_above_200dma'] & (df['dd_from_52w_high']<=-0.10),
}
for name,mask in comb_defs.items():
    r = evaluate(mask, name, 'Combination', gap=63)
    if r: results.append(r)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
res_df = pd.DataFrame(results)

# Ranking score: win rate, risk-adj return, low frequency, practicality
def rank_score(r):
    if r.get('category')=='Benchmark': return -999
    win = r.get('win_rate_6m',50)/100
    ret6 = r.get('avg_6m',0)/100
    dd = abs(r.get('avg_maxdd_12m',-20))/100
    ann = max(r.get('ann_occurrences',99),0.1)
    stab = 1.2 if r.get('stable') else 1.0
    sharpe_like = ret6/dd if dd>0 else 0
    freq_penalty = min(1, 3/ann)
    return (win*40 + min(sharpe_like,3)/3*30 + freq_penalty*20 + 10) * stab

ranked = res_df[res_df['category']!='Benchmark'].copy()
ranked['rank_score'] = ranked.apply(rank_score,axis=1)
ranked = ranked.sort_values('rank_score',ascending=False).reset_index(drop=True)
ranked.insert(0,'rank',range(1,len(ranked)+1))

res_df.to_csv('backtest_results.csv',index=False)
ranked.to_csv('ranked_strategies.csv',index=False)

# Save monthly seasonality detail for chart
monthly = []
for m in range(1,13):
    sub = df[df['month']==m]
    monthly.append({'month':pd.Timestamp(2020,m,1).month_name(),
                    'avg_6m':round(sub['fwd_6m'].mean()*100,2),
                    'win_6m':round((sub['fwd_6m']>0).mean()*100,1),
                    'avg_12m':round(sub['fwd_12m'].mean()*100,2)})
pd.DataFrame(monthly).to_csv('monthly_seasonality.csv',index=False)

# Save daily series for Opportunity Score backtest later
df.to_csv('daily_features.csv')

print(f"Sample: {df.index.min().date()} to {df.index.max().date()} ({len(df)} trading days)\n")
print("TOP 15 STRATEGIES BY RANK SCORE:")
print(ranked[['rank','name','category','ann_occurrences','win_rate_6m','avg_6m','avg_12m','avg_maxdd_12m','t_stat_6m','stable','rank_score']].head(15).to_string(index=False))
print("\nBASELINE:", baseline)
print("\nResults saved to backtest_results.csv, ranked_strategies.csv, monthly_seasonality.csv, daily_features.csv")
