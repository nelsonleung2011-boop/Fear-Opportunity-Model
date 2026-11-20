#!/usr/bin/env node
/**
 * Offline runtime harness for the v3 (dual-calibrated MOS) dashboard.
 * Executes the exact inline JS from index.html against a DOM stub and
 * ECharts stub, then calls init() and every render function to catch
 * runtime errors that `node --check` cannot detect.
 */
const fs = require('fs');
const path = require('path');

const HTML = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const scripts = [...HTML.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const code = scripts.join('\n');

let checks = 0, failures = 0;
function check(cond, msg) {
  checks++;
  if (!cond) { failures++; console.log('  FAIL: ' + msg); }
}

// ---- DOM stub ----
const elements = new Map();
function makeEl(id) {
  return {
    id, innerHTML: '', textContent: '', value: '', style: {},
    min: '', max: '', dataset: {},
    classList: { add(){}, remove(){}, contains(){return false;} },
    setAttribute(){}, getAttribute(){return null;},
    appendChild(){}, addEventListener(){},
  };
}
const documentStub = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeEl(id));
    return elements.get(id);
  },
  querySelector() { return makeEl('q'); },
  querySelectorAll() { return []; },
  documentElement: { setAttribute(){}, getAttribute(){return 'light';} },
};
const chartStub = { setOption(o){ chartCalls++; lastOption = o; }, resize(){}, dispose(){}, on(){} };
let chartCalls = 0, lastOption = null;
const echartsStub = { init(){ return chartStub; } };
const windowStub = {};
const localStorageStub = { getItem(){return null;}, setItem(){}, removeItem(){} };
const getComputedStyleStub = () => ({ getPropertyValue(){ return '#1f5fd0'; } });

// Extract embedded D payload from the code (the code declares `const D` itself)
const dm = /const D = (\{[\s\S]*?\});\s*const N = D\.n;/.exec(code);
check(!!dm, 'embedded D payload found');
const D = Function('return ' + dm[1])();

const factory = new Function(
  'document','window','localStorage','getComputedStyle','echarts','console',
  code + '\nreturn { init, nav, renderOverview, renderExplorer, renderCalibration, '
  + 'renderSectors, renderRegime, renderAudit, renderAllCharts, computeConfidence, '
  + 'renderDual, renderAnalogues, renderValidation, renderContributors, computeAction, '
  + 'mosBandOf, mosRowFor, bandOf, bandColor, calRowFor, lastValid, D, N };'
);

console.log('=== v3 Dashboard offline runtime harness ===');
check(D.n > 1000, 'payload has full history (n=' + D.n + ')');
check(D.sectors && D.sectors.sectors && D.sectors.sectors.length === 11, '11 sector ETFs present');
check(D.backtest && D.backtest.calibration && D.backtest.calibration.length >= 5, 'calibration bands present');
check(D.audit && D.audit.summary, 'audit summary present');
check(D.z_trend && D.comp_trend && D.comp_calm && D.comp_macro, 'explainability components present');
check(D.legacy && D.legacy.bands && D.legacy.bands.length >= 4, 'legacy comparison present');
check(D.mos && D.mos.length === D.n, 'MOS series present at full length');
check(D.backtest && D.backtest.dual_calibration && D.backtest.dual_calibration.length >= 5, 'dual calibration bands present');
check(D.validation && typeof D.validation.passed === 'boolean', 'validation report present');
check(D.contrib && Object.keys(D.contrib).length === 7, 'seven explainability contributors present');
check(D.analogues && D.analogues.length === 5, 'five historical analogues present');

const api = factory(documentStub, windowStub, localStorageStub, getComputedStyleStub, echartsStub, console);
check(typeof api.init === 'function', 'init function defined');

try {
  api.init();
  console.log('  init() executed without throwing');
} catch (e) {
  failures++;
  console.log('  FAIL: init() threw: ' + e.stack);
}

// Verify key outputs got populated
check((elements.get('ovKpis')||{}).innerHTML && elements.get('ovKpis').innerHTML.includes('Market Opportunity Score'), 'Decision Panel KPIs rendered');
check((elements.get('ovDecisionTable')||{}).innerHTML && elements.get('ovDecisionTable').innerHTML.includes('Expected 12M return'), 'Decision Panel table rendered');
check((elements.get('ovAction')||{}).innerHTML && elements.get('ovAction').innerHTML.includes('Why:'), 'Investment action guidance rendered');
check((elements.get('ovConfidence')||{}).innerHTML && elements.get('ovConfidence').innerHTML.includes('Confidence'), 'Confidence indicator rendered');
check((elements.get('dualTable')||{}).innerHTML && elements.get('dualTable').innerHTML.includes('Exp 12M'), 'Dual calibration table rendered');
check((elements.get('valChecks')||{}).innerHTML && elements.get('valChecks').innerHTML.includes('Monotonic expected return'), 'Validation checks rendered');
check((elements.get('anTable')||{}).innerHTML && elements.get('anTable').innerHTML.includes('Similarity'), 'Analogue table rendered');
check((elements.get('exBreakdown')||{}).innerHTML && elements.get('exBreakdown').innerHTML.includes('Contribution') || (elements.get('exBreakdown')||{}).innerHTML.includes('pts'), 'Score breakdown rendered');
check((elements.get('calTable')||{}).innerHTML && elements.get('calTable').innerHTML.includes('Excess 6M'), 'Calibration table rendered');
check((elements.get('secTable')||{}).innerHTML && elements.get('secTable').innerHTML.includes('Buying Score'), 'Sector table rendered');
check((elements.get('auditTable')||{}).innerHTML && elements.get('auditTable').innerHTML.includes('PASS'), 'Audit table rendered');
check((elements.get('legacyNote')||{}).innerHTML && elements.get('legacyNote').innerHTML.includes('redesign') || (elements.get('legacyNote')||{}).innerHTML.includes('rebuilt'), 'Legacy justification rendered');
check(chartCalls > 0, 'charts received setOption (' + chartCalls + ' calls)');

// Verify 6M win rate is materially higher in the top band than the bottom
const cal = D.backtest.calibration;
const bottomWin = cal[0].win_rate_6m, topWin = cal[cal.length-1].win_rate_6m;
check(topWin - bottomWin >= 20, 'top band win rate exceeds bottom by >=20pp: '
  + bottomWin + '% -> ' + topWin + '%');
check(topWin >= D.backtest.baseline_win_6m, 'top band beats unconditional baseline: '
  + topWin + '% vs ' + D.backtest.baseline_win_6m + '%');

// Verify the three highest bands all beat baseline (genuine high-zone edge)
const topBands = cal.slice(-3);
check(topBands.every(r => r.win_rate_6m >= D.backtest.baseline_win_6m),
  'all high bands beat baseline: ' + topBands.map(r=>r.win_rate_6m+'%').join(','));

// Verify components present for explainability
check(D.comp_trend && D.comp_calm && D.comp_macro, 'three components available for explainability');

// Dual-objective monotonicity: MOS bands must rise in win rate AND expected return
const dual = D.backtest.dual_calibration;
const rising = (a, tol) => a.every((v, i) => i === 0 || v >= a[i - 1] - tol);
check(rising(dual.map(r => r.win_rate_6m), 3), 'dual calibration win rate monotonic');
check(rising(dual.map(r => r.avg_6m), 1), 'dual calibration expected 6M return monotonic');
check(rising(dual.map(r => r.avg_12m), 1.5), 'dual calibration expected 12M return monotonic');
check(D.validation.passed, 'validation report passes');
check(D.validation.brier_score < 0.25, 'Brier score below 0.25');
check(dual[dual.length - 1].avg_6m > dual[0].avg_6m + 3, 'top MOS band beats bottom on expected return');
const bull = (elements.get('exBull')||{}).innerHTML || '';
check(bull.includes('pts') || bull.includes('+'), 'bullish contributors rendered');

console.log('\n' + (failures === 0 ? 'PASS' : 'FAIL') + ': ' + (checks - failures) + '/' + checks + ' checks passed');
process.exit(failures === 0 ? 0 : 1);
