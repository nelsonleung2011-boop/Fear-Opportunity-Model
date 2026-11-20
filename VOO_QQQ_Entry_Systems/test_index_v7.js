#!/usr/bin/env node
/** Offline runtime harness for index_v7.html -- exact inline JS + realistic DOM stub.
 *
 * The DOM stub deliberately returns null for ids that do not exist in the real
 * template. The previous harness fabricated every element, which is why it
 * passed while the tabs were broken in a browser. Tab buttons are registered
 * from the HTML that RENDER.tabs() actually produces.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const FILE = path.join(__dirname, 'index_v7.html');
const HTML = fs.readFileSync(FILE, 'utf8');
const pm = HTML.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/);
if (!pm) { console.error('FAIL: payload script missing'); process.exit(1); }
const scripts = [...HTML.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
if (!scripts.length) { console.error('FAIL: inline JS missing'); process.exit(1); }

let checks = 0, failures = 0;
function check(cond, msg) {
  checks++;
  if (!cond) { failures++; console.log('  FAIL: ' + msg); }
}

const REAL_IDS = new Set(['metastrip','hero','tabbar','sitefooter',
  'pane-decision','pane-regime','pane-credit','pane-internals','pane-cost',
  'pane-why','pane-data','pane-analogues','pane-ops','pane-method']);
const els = new Map();
function makeEl(id) {
  return { id, innerHTML: '', textContent: '', className: '', dataset: {}, style: {},
           listeners: {},
           addEventListener(ev, fn) { this.listeners[ev] = fn; },
           getElementsByTagName() { return []; },
           setAttribute() {}, appendChild() {} };
}
const payloadEl = { textContent: pm[1] };
const document = {
  body: { className: '' },
  listeners: {},
  addEventListener(ev, fn) { this.listeners[ev] = fn; },
  getElementById(id) {
    if (id === 'payload') return payloadEl;
    if (!REAL_IDS.has(id) && !els.has(id)) return null;
    if (!els.has(id)) els.set(id, makeEl(id));
    return els.get(id);
  }
};
const window = { scrollTo() {}, location: { hash: '' } };
const ctx = { document, window, console, JSON, Math, Number, String, Object,
              Array, Set, Date, RegExp, isNaN, parseFloat, parseInt };
ctx.__registerTab = function (name) {
  if (!els.has('tab-' + name)) els.set('tab-' + name, makeEl('tab-' + name));
};
vm.createContext(ctx);

const KEYS = ['meta','hero','tabs','decision','regime','credit','internals','cost',
              'why','data','analogues','ops','method','footer'];
const testCode = `
;(function(){
  globalThis.__results = {};
  for (const t of ${JSON.stringify(KEYS)}) {
    try { globalThis.__results[t] = RENDER[t](); }
    catch (e) { globalThis.__results[t] = 'ERROR: ' + e.stack; }
  }
  globalThis.__heroHTML = document.getElementById('hero').innerHTML;
  globalThis.__tabHTML = document.getElementById('tabbar').innerHTML;
  globalThis.__bootWhy = document.getElementById('pane-why').innerHTML;

  // Register exactly the buttons that RENDER.tabs() emitted, then simulate
  // a genuine click bubbling to the persistent tabbar container.
  for (const m of globalThis.__tabHTML.matchAll(/id="tab-([^"]+)"/g)) {
    globalThis.__registerTab(m[1]);
  }
  const bar = document.getElementById('tabbar');
  globalThis.__hasDelegatedTab = typeof bar.listeners.click === 'function';
  globalThis.__hasDocDelegation = typeof document.listeners.click === 'function';
  bar.listeners.click({ target: { getAttribute: k => k === 'data-tab' ? 'ops' : null } });
  globalThis.__afterClick = {
    active: activeTab,
    paneOps: document.getElementById('pane-ops').className,
    paneDecision: document.getElementById('pane-decision').className,
    btnOps: document.getElementById('tab-ops').className,
    btnDecision: document.getElementById('tab-decision').className
  };
  bar.listeners.click({ target: { getAttribute: k => k === 'data-tab' ? 'data' : null } });
  globalThis.__afterData = {
    active: activeTab,
    paneData: document.getElementById('pane-data').className,
    paneOps: document.getElementById('pane-ops').className
  };
  setTab('why');
  globalThis.__afterTab = {
    active: activeTab,
    paneClass: document.getElementById('pane-why').className,
    decisionClass: document.getElementById('pane-decision').className,
    btnClass: document.getElementById('tab-why').className
  };

  // The mode switch is delegated from the persistent pane, so it must keep
  // working after the pane's innerHTML has been rebuilt several times.
  const whyPane = document.getElementById('pane-why');
  globalThis.__hasModeDelegation = typeof whyPane.listeners.click === 'function';
  function modeClick(mode){
    whyPane.listeners.click({ target: { getAttribute: k => k === 'data-mode' ? mode : null } });
  }
  modeClick('expert'); globalThis.__expertBody = document.body.className;
  modeClick('simple'); globalThis.__simpleBody = document.body.className;
  modeClick('expert'); globalThis.__expertAgain = document.body.className;
  modeClick('simple'); globalThis.__simpleAgain = document.body.className;
})();
`;
try {
  vm.runInContext(scripts.join('\n') + testCode, ctx, { filename: 'index_v7.inline.js' });
} catch (e) {
  console.error('FAIL: top-level JS threw: ' + e.stack);
  process.exit(1);
}

const R = ctx.__results;
check(Object.keys(R).length === KEYS.length, 'all renderers registered');
for (const k of KEYS) {
  check(typeof R[k] === 'string' && !R[k].startsWith('ERROR:'), 'renderer ' + k + ' produced HTML');
}
const D = JSON.parse(pm[1]);
const E = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
  .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
check(!!D.ops && !!D.data_audit, 'operational + provenance payloads embedded');

// hero answers all four concepts
for (const word of ['Market Regime','Valuation','Opportunity','Deployment']) {
  check(ctx.__heroHTML.includes(word), 'hero shows ' + word);
}
check(ctx.__heroHTML.includes(D.hero.reason.slice(0, 20)), 'hero shows reason');
check(ctx.__heroHTML.includes(D.hero.deployment), 'hero shows deployment value');

// ---- THE TAB BUG: real ids + delegated click must actually switch panes ----
for (const t of ['decision','regime','credit','internals','cost','why','data','analogues','ops','method']) {
  check(ctx.__tabHTML.includes('id="tab-' + t + '"'), 'tab button has real id: ' + t);
  check(ctx.__tabHTML.includes('data-tab="' + t + '"'), 'tab button has data-tab: ' + t);
}
check(ctx.__hasDelegatedTab, 'tabbar has a delegated click listener');
check(ctx.__hasDocDelegation, 'document has a delegated click listener');
check(ctx.__afterClick.active === 'ops', 'real click switches activeTab to ops');
check(ctx.__afterClick.paneOps.includes('active'), 'real click activates ops pane');
check(!ctx.__afterClick.paneDecision.includes('active'), 'real click deactivates decision pane');
check(ctx.__afterClick.btnOps === 'active', 'real click highlights ops button');
check(ctx.__afterClick.btnDecision === '', 'real click clears decision button');
check(ctx.__afterData.active === 'data' && ctx.__afterData.paneData.includes('active'),
      'second real click switches to data tab');
check(!ctx.__afterData.paneOps.includes('active'), 'second click deactivates previous pane');
check(ctx.__afterTab.active === 'why' && ctx.__afterTab.paneClass.includes('active'),
      'setTab switches active tab directly');
check(!ctx.__afterTab.decisionClass.includes('active'), 'setTab deactivates decision');
check(ctx.__afterTab.btnClass === 'active', 'setTab highlights the button');

// ---- mode switch must survive pane rebuilds ----
check(ctx.__hasModeDelegation, 'why pane has delegated mode listener');
check(ctx.__expertBody === 'expert', 'expert mode adds body class');
check(ctx.__simpleBody === '', 'simple mode clears body class');
check(ctx.__expertAgain === 'expert', 'expert mode still works after a rebuild');
check(ctx.__simpleAgain === '', 'simple mode still works after two rebuilds');

// decision panel
check(R.decision.includes('VOO BUYING SCORE') && R.decision.includes('QQQ BUYING SCORE'),
      'decision shows VOO and QQQ');
check(R.decision.includes('DEPLOYMENT LADDER'), 'deployment ladder present');
check(R.decision.includes('Historical percentile'), 'decision shows percentile ranking');
for (const s of D.sensitivity) {
  const escaped = s.scenario.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  check(R.decision.includes(escaped), 'sensitivity scenario shown: ' + s.scenario);
}

// regime / credit / internals / cost
check(R.regime.includes(D.regime.state), 'regime state rendered');
check(R.regime.includes(D.regime.confidence), 'regime confidence rendered');
for (const k of Object.keys(D.regime.inputs)) check(R.regime.includes(k), 'regime input: ' + k);
for (const k of Object.keys(D.credit.components)) check(R.credit.includes(k.replace(/_/g,' ')), 'credit input: ' + k);
for (const k of Object.keys(D.internals.components)) check(R.internals.includes(k.replace(/_/g,' ')), 'internals input: ' + k);
check(R.cost.includes('Is waiting for a correction'), 'opportunity cost question');
for (const s of D.instruments.VOO.opportunity_cost.scenarios)
  check(R.cost.includes(s.correction_pct + '% correction'), 'scenario ' + s.correction_pct + '%');

// WHY THIS SCORE (mandatory explainability)
check(R.why.includes('Why this score?'), 'why heading');
check(R.why.includes('Score Audit Trail'), 'audit trail present');
check(R.why.includes('FORMULA DISCLOSURE'), 'formula disclosure present');
check(R.why.includes(E(D.instruments.VOO.explain.reconciliation)),
      'base + bonus reconciliation shown');
check(R.why.includes('VOO Buying Score = ' + D.instruments.VOO.score.toFixed(1)),
      'why tab shows the final score, not the base score');
check(R.why.includes('Biggest positive contributors'), 'top positive present');
check(R.why.includes('Biggest negative contributors'), 'top negative present');
check(R.why.includes('Simple Mode') && R.why.includes('Expert Mode'), 'mode switch present');
for (const c of D.instruments.VOO.explain.components) {
  check(R.why.includes(c.name), 'explain contributor: ' + c.name);
  check(R.why.includes(String((c.weight*100).toFixed(1))), 'explain weight: ' + c.name);
}
for (const a of D.instruments.VOO.audit) check(R.why.includes(a.step), 'audit step: ' + a.step);
for (const k of Object.keys(D.formula_disclosure))
  check(R.why.includes(D.formula_disclosure[k].definition.slice(0, 15)), 'formula definition: ' + k);

// DATA & SOURCES tab
check(R.data.includes('Is the data legit?'), 'data tab verdict heading');
check(R.data.includes('SCORE INPUT LEDGER'), 'score input ledger present');
check(R.data.includes('SOURCE INVENTORY'), 'source inventory present');
check(R.data.includes('RAW PRICE TAPE'), 'raw price snapshot present');
for (const s of D.data_audit.sources)
  check(R.data.includes(E(s.name)), 'source listed: ' + s.name);
for (const c of D.data_audit.score_inputs)
  check(R.data.includes(E(c.name)), 'score input listed: ' + c.name);
for (const v of D.data_audit.verdict)
  check(R.data.includes(v.level), 'verdict level shown: ' + v.level);
check(R.data.includes('LATEST REVISED'), 'revised-vintage caveat disclosed');
check(R.data.includes('hard-coded 2026-11-16'), 'frozen as-of fix disclosed');
for (const d of D.data_audit.price_snapshot.dates.slice(0, 3))
  check(R.data.includes(d), 'price snapshot date shown: ' + d);

// OPERATIONS tab
check(R.ops.includes('ALERT RULES'), 'alert rules present');
check(R.ops.includes('SCORE HISTORY') || R.ops.includes('Score history'), 'score history present');
check(R.ops.includes('ALERT LOG'), 'alert log present');
check(R.ops.includes('SETUP'), 'setup instructions present');
check(R.ops.includes('Weekly summary'), 'weekly summary rule present');
check(R.ops.includes(E('VOO >= 80')) && R.ops.includes(E('QQQ >= 80')), '80 thresholds shown');
for (const r of D.ops.rules) check(R.ops.includes(E(r.name)), 'rule shown: ' + r.name);
check(R.ops.includes(D.ops.setup.command), 'daily command shown');
for (const a of D.ops.recent_alerts) check(R.ops.includes(a.type), 'alert log type: ' + a.type);
check(R.ops.includes('no daily spam') || R.ops.includes('send nothing'), 'no-spam policy disclosed');

// analogues + footer
check((R.analogues.match(/simcard/g) || []).length === D.analogues.length, 'all analogues rendered');
check(R.footer.includes(D.meta.version), 'footer version');

// forbidden concepts must not be presented as product features
for (const f of ['Timing Advantage','Evidence Insufficient','Predicted 12M Return',
                 'XGBoost','Random Forest','AUC']) {
  check(!HTML.includes(f), 'v7 omits: ' + f);
}

console.log('v7 dashboard JS: ' + checks + ' checks, ' + failures + ' failures');
if (failures) process.exit(1);
