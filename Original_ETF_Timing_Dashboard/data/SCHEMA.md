# dashboard_data.json — schema v4

One JSON payload embedded into `index.html` by `build_dashboard.py`.
Series arrays are length `n`, index-aligned with `dates`; missing values are `null`.

## meta
`generated_at`, `as_of`, `n`, `start`, `schema_version` (4), `model_version`,
`horizon_sessions` (126), `embargo_sessions` (126), `price_basis`
("dividend-adjusted" | "raw close"), `providers` (per symbol), `stale_days`,
`disclaimer`.

## top-level series
| field | meaning |
|---|---|
| `prob` | calibrated P(positive 6M total return), 0-100. Natural units, never rescaled. `null` before the first walk-forward origin. |
| `mu` | expected 6M total return, %, ridge regression, out of sample |
| `median6` | median 6M return implied by training residuals |
| `ploss10` | P(6M return < -10%) from training residuals |
| `opp` | Opportunity Score 0-100 |
| `timing_10y`, `timing_5y`, `timing_exp` | point-in-time percentile of `opp` within trailing 1260 / 2520 / expanding sessions |
| `fam` | five family percentiles: `return_expectation`, `dislocation`, `downside_risk`, `valuation_trend`, `internals` |
| `comp` | raw component percentiles behind the families |
| `spy`, `vix`, `dd`, `vixpct`, `sma200`, `high52`, `trend_dev` | causal inputs for charting |
| `regime` | Risk-On / Neutral / Risk-Off |

## current
Snapshot for `as_of`: `prob_pct`, `prob_bin` (predicted, observed, n_raw, n_eff,
ci_low, ci_high, range), `calibration_method`, `ece_pp`, `calibration_warning`,
`exp_ret_pct`, `median_ret_pct`, `p_loss10_pct`, `es6_pct`, `opp`, `opp_band`,
`timing`, `families`, `weights`, `regime`, `conviction` (grade A-D, score,
reasons, ci_low/high, allow_action_language), `matrix` (quadrant, headline,
detail), `stale_days`, `spy`, `vix`, `drawdown_pct`.

## calibration
Out-of-sample diagnostics on the concatenated test blocks: `n`, `n_eff`,
`brier`, `brier_baseline` (constant forecast of the OOS base rate),
`log_loss`, `log_loss_baseline`, `ece_pp`, `mce_pp`, `slope`, `intercept`,
`base_rate`, `reliability` (bins with `n_eff`, low-evidence bins merged),
`n_independent` (non-overlapping monthly count), `brier_monthly`,
`by_decade`, `by_regime`, `by_vol_tercile` (each with a within-group
calibration slope), `methods_comparison` (identity/platt/isotonic/beta scored
on identical purged walk-forwards).

## blocks
One entry per walk-forward origin: `origin_index`, `val_start_index`,
`train_last_index`, `origin`, `test_end`, `train_rows`, `train_last_date`,
`purged_rows`, `val_rows`, `purge_gap_sessions`, `calibrator`,
`cv_scores`, `base_rate_train`, `anchor_at_test`.

## signal_tests
`window` (start, end, n_non_overlapping), `base_win_pct`, `tests[]` with
`signal`, `hi_win_pct`, `lo_win_pct`, `gap_pp`, `t`, `p_value`, `significant`,
plus `conclusion` and a multiple-testing note.

## benchmarks
`window`, `assumptions`, `strategies[]` with `terminal_wealth`, `cagr_pct`,
`vol_pct`, `sharpe`, `sortino`, `max_drawdown_pct`, `calmar`,
`time_invested_pct`, `time_in_cash_pct`, `buys`, `worst_5pct_12m`,
`regret_vs_lump_sum`, `turnover`; plus a regime-matched random-strategy mean
over 200 draws.

## analogues
`target_date`, `features_used`, `features_dropped` (collinear removals),
`episodes_found`, `episode_gap_sessions`, `rows[]` (date, episode, distance,
similarity_pct, fwd_1m/3m/6m/12m, drivers), `stats` (median/mean/p25/p75/worst/
best per horizon), `dispersion_6m_iqr`, `disagreement_warning`, `disclaimer`.

## audit_method / audit_data
`audit_method.items[]` = `{item, status: PASS|WARNING|FAIL, finding}` covering
methodology; `audit_data` is the per-symbol download audit from
`validate_data.py`. They are reported on separate tabs on purpose.

## sensitivity
Alternative Opportunity weightings with `spearman_vs_base`, `mean_abs_diff`
and `label_change_pct`.
