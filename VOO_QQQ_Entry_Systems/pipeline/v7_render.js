/* v7.1 renderer — pure builders, tab controller, Simple/Expert switch, ops panes.
 *
 * Navigation fix: every tab button carries a real id AND clicks are delegated
 * from the persistent #tabbar container. The WHY pane is rebuilt on a mode
 * switch, so its switch is delegated from the persistent #pane-why container.
 * This is why the original v7 tabs did nothing in a real browser.
 */
const D = JSON.parse(document.getElementById('payload').textContent);
let activeTab = 'decision';
let expert = false;
const TABS = [
  ['decision','Decision'],
  ['regime','Market Regime'],
  ['credit','Credit Stress'],
  ['internals','Market Internals'],
  ['cost','Opportunity Cost'],
  ['why','WHY THIS SCORE?'],
  ['data','DATA &amp; SOURCES'],
  ['analogues','Historical Analogues'],
  ['ops','OPERATIONS'],
  ['method','Method']
];

function esc(v){
  if (v === null || v === undefined) return '';
  return String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function num(v,d){
  if (v === null || v === undefined || isNaN(v)) return 'n/a';
  return Number(v).toFixed(d === undefined ? 0 : d);
}
function colorFor(s){
  if (s >= 90) return '#17745a';
  if (s >= 60) return '#2b8c6e';
  if (s >= 40) return '#1d6f8c';
  if (s >= 20) return '#a26616';
  return '#ad342d';
}
function scoreBar(s){
  return '<div class="scorebar"><i style="width:' + Number(s) + '%;background:' + colorFor(s) + '"></i></div>';
}
function badge(state){
  const s = String(state);
  let cls = 'no';
  if (s === 'YES' || s === 'BULL' || s === 'High' || s === 'OK' || s === 'REAL') cls = 'ok';
  else if (s === 'WATCHLIST' || s === 'CORRECTION' || s === 'Moderate' || s === 'CAVEAT'
           || s === 'PROXY' || s === 'CONSTRUCTED' || s === 'FIXED') cls = 'mid';
  else if (s === 'BEAR' || s === 'PANIC' || s === 'VERDICT') cls = 'bear';
  return '<span class="badge ' + cls + '">' + esc(s) + '</span>';
}
function kv(label, value){
  return '<div class="wrow"><span class="wname">' + esc(label) + '</span><span class="wval">'
    + esc(value) + '</span></div>';
}
function table(head, rows){
  return '<div class="scroll"><table><thead><tr>' + head.map(function(h){
    return '<th>' + h + '</th>'; }).join('') + '</tr></thead><tbody>' + rows.join('')
    + '</tbody></table></div>';
}

const RENDER = {};

RENDER.meta = function(){
  const m = D.meta;
  let h = '<span class="pill key">AS OF ' + esc(m.as_of) + '</span>'
    + '<span class="pill">4 SEPARATE CONCEPTS</span>'
    + '<span class="pill">NOT A FORECAST ENGINE</span>';
  const o = D.ops || {};
  if (o.status && o.status.last_run) h += '<span class="pill">DAILY RUN ' + esc(o.status.last_run.slice(0,10)) + '</span>';
  return h;
};

RENDER.hero = function(){
  const h = D.hero;
  return '<div class="hero"><div class="hgrid">'
    + '<div class="hcell"><div class="hlabel">Market Regime</div>'
      + '<div class="hval">' + esc(h.market_regime) + '</div>'
      + '<div class="hsub">' + esc(h.regime_confidence) + ' confidence</div></div>'
    + '<div class="hcell"><div class="hlabel">Valuation</div>'
      + '<div class="hval">' + esc(h.valuation) + '</div>'
      + '<div class="hsub">' + num(D.valuation.score,1) + '/100</div></div>'
    + '<div class="hcell"><div class="hlabel">Opportunity</div>'
      + '<div class="hval">' + esc(h.opportunity) + '</div>'
      + '<div class="hsub">' + num(D.opportunity.score,1) + '/100</div></div>'
    + '<div class="hcell"><div class="hlabel">Deployment</div>'
      + '<div class="hval deploy">' + esc(h.deployment) + '</div>'
      + '<div class="hsub">' + esc(h.deployment_label) + '</div></div>'
    + '</div><div class="reason"><b>Reason:</b> ' + esc(h.reason) + '</div></div>';
};

RENDER.tabs = function(){
  return TABS.map(function(t){
    return '<button type="button" id="tab-' + t[0] + '" data-tab="' + t[0] + '" class="'
      + (t[0]===activeTab?'active':'') + '">' + t[1] + '</button>';
  }).join('');
};

RENDER.decision = function(){
  const v = D.instruments.VOO, q = D.instruments.QQQ;
  let h = '<section><div class="sechead"><span class="secnum">DEPLOYMENT</span>'
    + '<h2>Cash deployment guidance</h2>'
    + '<span class="sechnote">Purchase pacing only — never a market-direction call</span></div><div class="qgrid3">';
  [['VOO',v],['QQQ',q]].forEach(function(pair){
    const name = pair[0], x = pair[1], dep = x.deployment;
    const pct = D.ops && D.ops.percentiles ? D.ops.percentiles[name] : null;
    h += '<div class="qcard"><div class="qlabel">' + name + ' BUYING SCORE</div>'
      + '<div class="qans">' + num(x.score,1) + '/100</div>'
      + '<div class="qlab2">' + esc(x.label) + '</div>' + scoreBar(x.score)
      + '<div class="qdet"><b>Deploy ' + dep.percent + '%:</b> ' + esc(dep.label) + '<br>'
      + 'Valuation ' + num(D.valuation.score,1) + ' · Opportunity ' + num(D.opportunity.score,1)
      + ' · Risk ' + esc(D.risk.meter) + ' · bonus ' + num(x.drawdown_bonus,1) + '<br>'
      + 'Historical percentile ' + (pct===null?'n/a':num(pct,1) + 'th') + '<br>'
      + 'Rare detector ' + badge(x.detector.state) + '</div></div>';
  });
  h += '<div class="qcard"><div class="qlabel">DEPLOYMENT LADDER</div>'
    + '<div style="margin-top:10px">'
    + [[90,'100%','Rare opportunity'],[80,'75%','Aggressive'],[60,'50%','Moderate'],
       [40,'25%','Small'],[0,'0%','Pause lump sum']].map(function(r){
        return '<div class="wrow"><span class="wname">' + r[2] + '</span><span class="wval">' + r[1] + '</span></div>';
      }).join('') + '</div></div></div>';
  h += '</section><section><div class="sechead"><span class="secnum">WHAT WOULD CHANGE IT</span>'
    + '<h2>Sensitivity — if conditions deteriorated</h2></div><div class="card"><div class="scroll"><table>'
    + '<thead><tr><th>Scenario</th><th>Score</th><th>Change</th><th>Deployment</th><th>Band</th></tr></thead><tbody>';
  D.sensitivity.forEach(function(s){
    h += '<tr><td>' + esc(s.scenario) + '</td><td>' + num(s.score,1) + '</td><td>+'
      + num(s.change,1) + '</td><td>' + s.deployment_pct + '%</td><td>' + esc(s.label) + '</td></tr>';
  });
  h += '</tbody></table></div><p class="note" style="margin-top:10px">Each scenario changes only the '
    + 'named inputs and holds everything else at today\'s values. These are mechanical '
    + 'recomputations, not forecasts.</p></div></section>';
  return h;
};

RENDER.regime = function(){
  const r = D.regime;
  let h = '<section><div class="sechead"><span class="secnum">TOP PANEL</span>'
    + '<h2>Market Regime — ' + esc(r.state) + ' (' + esc(r.confidence) + ' confidence)</h2></div>'
    + '<div class="qgrid4">';
  Object.keys(r.inputs).forEach(function(k){
    const x = r.inputs[k];
    h += '<div class="qcard"><div class="qlabel">' + esc(k) + '</div>'
      + '<div class="qans small">' + (typeof x.raw === 'number' ? num(x.raw, x.raw < 1 ? 3 : 1) : esc(x.raw))
      + '</div><div class="qlab2">' + badge(x.state) + '</div>'
      + '<div class="qdet">' + esc(x.unit || '') + '</div></div>';
  });
  h += '</div><div style="height:14px"></div><div class="card">'
    + '<div class="wrow"><span class="wname">Vote distribution</span><span class="wval">'
    + Object.keys(r.votes).map(function(k){ return k + ' ' + r.votes[k]; }).join(' · ')
    + '</span></div><div class="wrow"><span class="wname">Agreement</span><span class="wval">'
    + num(r.agreement_pct,0) + '%</span></div>'
    + '<p class="note" style="margin-top:10px">' + esc(r.rule) + '</p></div></section>';
  return h;
};

RENDER.credit = function(){
  const c = D.credit;
  let h = '<section><div class="sechead"><span class="secnum">INDEPENDENT LAYER</span>'
    + '<h2>Credit Stress Score — ' + num(c.score,1) + '/100 (' + esc(c.label) + ')</h2></div>'
    + '<div class="card"><div class="scroll"><table><thead><tr><th>Input</th><th>Raw</th>'
    + '<th>Normalised</th><th>Weight</th><th>Contribution</th><th>Note</th></tr></thead><tbody>';
  Object.keys(c.components).forEach(function(k){
    const x = c.components[k];
    const raw = (x.raw && typeof x.raw === 'object') ? JSON.stringify(x.raw) :
                (x.raw === null ? 'n/a' : (typeof x.raw === 'number' ? num(x.raw,3) : esc(x.raw)));
    h += '<tr><td>' + esc(k.replace(/_/g,' ')) + '</td><td>' + raw + '</td><td>'
      + num(x.normalised,1) + '</td><td>' + Math.round(x.weight*100) + '%</td><td>'
      + num(x.contribution,1) + '</td><td class="lt">' + esc(x.note) + '</td></tr>';
  });
  h += '</tbody></table></div><p class="note" style="margin-top:10px">' + esc(c.note)
    + '</p></div></section>';
  return h;
};

RENDER.internals = function(){
  const x = D.internals;
  let h = '<section><div class="sechead"><span class="secnum">HIDDEN WEAKNESS</span>'
    + '<h2>Market Internals — ' + esc(x.label) + '</h2></div>'
    + '<div class="card"><div class="scroll"><table><thead><tr><th>Input</th><th>Raw</th>'
    + '<th>Weakness score</th><th>Weight</th><th>Contribution</th><th>Definition</th></tr></thead><tbody>';
  Object.keys(x.components).forEach(function(k){
    const c = x.components[k];
    h += '<tr><td>' + esc(k.replace(/_/g,' ')) + '</td><td>' + num(c.raw,2) + ' ' + esc(c.unit||'')
      + '</td><td>' + num(c.normalised,1) + '</td><td>' + Math.round(c.weight*100) + '%</td><td>'
      + num(c.contribution,1) + '</td><td class="lt">' + esc(c.note) + '</td></tr>';
  });
  h += '</tbody></table></div><p class="note" style="margin-top:10px">' + esc(x.note)
    + '</p></div></section>';
  return h;
};

RENDER.cost = function(){
  const c = D.instruments.VOO.opportunity_cost;
  let h = '<section><div class="sechead"><span class="secnum">OPPORTUNITY COST</span>'
    + '<h2>Is waiting for a correction historically beneficial?</h2></div><div class="card">'
    + '<p class="note">Current VOO Buying Score <b>' + num(c.current_score,1)
    + '</b> at a ' + num(c.current_drawdown_pct,1) + '% drawdown. The scenarios below replace '
    + 'only the drawdown input and hold all other conditions at today\'s values.</p>'
    + '<div class="scroll" style="margin-top:12px"><table><thead><tr><th>Wait for</th>'
    + '<th>Status</th><th>Score then</th><th>Improvement</th><th>Hist. frequency</th>'
    + '<th>Median next 12M</th><th>n</th></tr></thead><tbody>';
  c.scenarios.forEach(function(s){
    h += '<tr><td>' + s.correction_pct + '% correction</td><td class="lt">'
      + esc(s.status || 'wait required') + '</td><td>' + num(s.score_at_scenario,1)
      + '</td><td>+' + num(s.score_improvement,1) + '</td><td>' + num(s.historical_freq_pct,1)
      + '%</td><td>' + (s.median_next_12m_pct===null?'n/a':num(s.median_next_12m_pct,1)+'%')
      + '</td><td>' + s.n_first_crossings + '</td></tr>';
  });
  h += '</tbody></table></div><p class="note" style="margin-top:12px">' + esc(c.note)
    + '</p></div></section>';
  return h;
};

function explainTable(title, ex){
  let h = '<h3>' + esc(title) + ' = ' + num(ex.final,1) + '</h3>'
    + '<div class="formula">' + esc(ex.formula) + '</div>'
    + '<div class="scroll" style="margin-top:10px"><table><thead><tr><th>Contributor</th>'
    + '<th>Raw</th><th>Normalised</th><th>Weight</th><th>Final contribution</th>'
    + '<th class="expert-only">Δ vs neutral</th></tr></thead><tbody>';
  ex.components.forEach(function(c){
    const raw = (c.raw !== null && typeof c.raw === 'object') ? JSON.stringify(c.raw)
              : (c.raw === null ? 'n/a' : (typeof c.raw === 'number' ? num(c.raw,3) : esc(c.raw)));
    h += '<tr><td>' + esc(c.name) + '</td><td>' + raw + '</td><td>' + num(c.normalised,1)
      + '</td><td>' + (c.weight*100).toFixed(1) + '%</td><td>' + num(c.contribution,2)
      + '</td><td class="expert-only">' + (c.delta_vs_neutral>=0?'+':'') + num(c.delta_vs_neutral,2)
      + '</td></tr>';
  });
  h += '</tbody></table></div>';
  h += '<div class="formula" style="margin-top:10px">' + esc(ex.reconciliation) + '</div>';
  h += '<div class="two" style="margin-top:12px"><div><b>Biggest positive contributors</b><ol class="clean">'
    + ex.top_positive.map(function(p){ return '<li>' + esc(p.name) + ' (+' + num(p.contribution,1)
      + ' pts)</li>'; }).join('') + '</ol></div><div><b>Biggest negative contributors</b><ol class="clean">'
    + ex.top_negative.map(function(p){ return '<li>' + esc(p.name) + ' (' + num(p.contribution,1)
      + ' pts; Δ ' + (p.delta_vs_neutral>=0?'+':'') + num(p.delta_vs_neutral,1) + ')</li>'; }).join('')
    + '</ol></div></div>';
  return h;
}

RENDER.why = function(){
  const v = D.instruments.VOO;
  let h = '<section><div class="sechead"><span class="secnum">EXPLAINABILITY</span>'
    + '<h2>Why this score?</h2><span class="sechnote">Every point is traceable</span></div>'
    + '<div class="card"><div class="switch" id="modeSwitch">'
    + '<button type="button" data-mode="simple" class="' + (!expert?'active':'') + '">Simple Mode</button>'
    + '<button type="button" data-mode="expert" class="' + (expert?'active':'') + '">Expert Mode</button></div>'
    + '<p class="note" style="margin-top:10px">Expert Mode reveals raw values, percentiles, '
    + 'transformations, weighting, intermediate calculations and the final formula. '
    + 'No black-box calculations.</p></div></section>';

  h += '<section><div class="card">' + explainTable('VOO Buying Score', v.explain)
    + '<h3>Sensitivity — what would improve the score?</h3><div class="scroll"><table><thead><tr>'
    + '<th>Change</th><th>New score</th><th>Change</th><th>Deployment</th></tr></thead><tbody>'
    + D.sensitivity.map(function(s){ return '<tr><td>' + esc(s.scenario) + '</td><td>'
      + num(s.score,1) + '</td><td>+' + num(s.change,1) + '</td><td>' + s.deployment_pct
      + '%</td></tr>'; }).join('') + '</tbody></table></div></div></section>';

  h += '<section><div class="card">' + explainTable('QQQ Buying Score', D.instruments.QQQ.explain)
    + '</div></section>';
  h += '<section><div class="two"><div class="card">' + explainTable('Valuation Score', D.valuation.explain)
    + '</div><div class="card">' + explainTable('Opportunity Score', D.opportunity.explain)
    + '</div></div></section>';

  h += '<section><div class="sechead"><span class="secnum">AUDIT TRAIL</span>'
    + '<h2>Score Audit Trail — reproduce it manually</h2></div><div class="card">'
    + '<div class="scroll"><table><thead><tr><th>Step</th><th>Value</th><th>Operation</th>'
    + '<th>Result</th></tr></thead><tbody>'
    + v.audit.map(function(a){ return '<tr><td>' + esc(a.step) + '</td><td>' + esc(a.value)
      + '</td><td>' + esc(a.operation) + '</td><td class="audit-num">' + num(a.result,1)
      + '</td></tr>'; }).join('') + '</tbody></table></div>'
    + '<p class="note" style="margin-top:10px">Valuation ' + num(D.valuation.score,1)
      + ' × 30% + Opportunity ' + num(D.opportunity.score,1) + ' × 50% + Risk '
      + num(D.risk.opportunity_score,0) + ' × 20% + bonus ' + num(v.drawdown_bonus,1)
      + ' = <b>' + num(v.score,1) + '</b>. The weighted layer components are listed '
      + 'in the VOO table above.</p></div></section>';

  h += '<section><div class="sechead"><span class="secnum">FORMULA DISCLOSURE</span>'
    + '<h2>Every indicator: definition, formula, direction, reason</h2></div>'
    + '<div class="qgrid3">';
  Object.keys(D.formula_disclosure).forEach(function(k){
    const f = D.formula_disclosure[k];
    const c = D.opportunity.components[k] || {};
    h += '<div class="qcard"><div class="qlabel">' + esc(k.replace(/_/g,' ')) + '</div>'
      + '<div class="kv" style="margin-top:8px"><b>Definition:</b> ' + esc(f.definition) + '</div>'
      + '<div class="kv"><b>Formula:</b></div><div class="formula">' + esc(f.formula) + '</div>'
      + '<div class="kv"><b>Direction:</b> ' + esc(f.direction) + '</div>'
      + '<div class="kv"><b>Reason:</b> ' + esc(f.reason) + '</div>'
      + '<div class="kv"><b>Example:</b> ' + esc(f.example) + '</div>'
      + '<div class="kv"><b>Current:</b> ' + (c.raw===null?'n/a':esc(c.raw))
      + ' · <b>Contribution:</b> ' + (c.score===undefined?'n/a':num(c.score,1)) + '</div></div>';
  });
  h += '</div></section>';
  return h;
};

/* ---------------------------------------------------------------------------
 * DATA & SOURCES — every number that forms the score, with provenance.
 * ------------------------------------------------------------------------- */
RENDER.data = function(){
  const A = D.data_audit || {};
  const S = A.sources || [], I = A.score_inputs || [], V = A.verdict || [];
  const sum = A.summary || {};
  let h = '<section><div class="sechead"><span class="secnum">DATA AUDIT</span>'
    + '<h2>Is the data legit?</h2><span class="sechnote">As of ' + esc(A.as_of || D.meta.as_of) + '</span></div>'
    + '<div class="qgrid4">'
    + '<div class="qcard"><div class="qlabel">INPUT SERIES</div><div class="qans">'
      + num(sum.n_sources,0) + '</div><div class="qdet">distinct source series feeding the score</div></div>'
    + '<div class="qcard"><div class="qlabel">DISCLOSED PROXIES</div><div class="qans">'
      + num(sum.n_proxies,0) + '</div><div class="qdet">no exchange breadth / options tape / market cap feed</div></div>'
    + '<div class="qcard"><div class="qlabel">OLDEST EFFECTIVE INPUT</div><div class="qans small">'
      + num(sum.max_staleness_days,0) + ' d</div><div class="qdet">age of the stalest value actually knowable on the scoring date</div></div>'
    + '<div class="qcard"><div class="qlabel">FABRICATED SERIES</div><div class="qans">0</div>'
      + '<div class="qdet">unavailable data is never invented; it is flagged and proxied</div></div>'
    + '</div></section>';

  h += '<section><div class="card"><h3>Verdict</h3><ul class="clean">'
    + V.map(function(x){ return '<li><b>' + esc(x.level) + ':</b> ' + esc(x.text) + '</li>'; }).join('')
    + '</ul></div></section>';

  h += '<section><div class="sechead"><span class="secnum">SCORE INPUT LEDGER</span>'
    + '<h2>Every input that forms the VOO Buying Score</h2></div><div class="card">'
    + table(['Input','Raw value','Layer','Transform','Normalised','Weight','Final pts','Source'],
      I.map(function(x){
        return '<tr><td>' + esc(x.name) + '</td><td>' + esc(x.raw_display) + '</td><td>'
          + esc(x.layer) + '</td><td class="lt">' + esc(x.transform) + '</td><td>'
          + num(x.normalised,1) + '</td><td>' + num(x.weight_pct,1) + '%</td><td class="audit-num">'
          + num(x.contribution,2) + '</td><td class="lt">' + esc(x.source) + '</td></tr>';
      }))
    + '<p class="note" style="margin-top:10px">Final points are the exact values used by the '
    + 'audit trail on the WHY THIS SCORE tab. Sum of layer contributions plus the drawdown '
    + 'bonus reproduces the displayed score.</p></div></section>';

  h += '<section><div class="sechead"><span class="secnum">SOURCE INVENTORY</span>'
    + '<h2>Where each series comes from</h2></div><div class="card">'
    + table(['Series','Source','Kind','Coverage','Last obs','Lag (d)','Knowable from','Stale (d)','Vintage','Role'],
      S.map(function(s){
        return '<tr><td>' + esc(s.name) + '</td><td class="lt">' + esc(s.source) + '</td><td>'
          + badge(s.kind) + '</td><td>' + esc(s.coverage) + '</td><td>' + esc(s.last_obs)
          + '</td><td>' + num(s.lag_days,0) + '</td><td>' + esc(s.effective_known_date)
          + '</td><td>' + num(s.staleness_days,0) + '</td><td class="lt">' + esc(s.vintage)
          + '</td><td class="lt">' + esc(s.role) + '</td></tr>';
      }))
    + '</div></section>';

  const snap = A.price_snapshot || {};
  const dates = snap.dates || [], series = snap.series || {};
  h += '<section><div class="sechead"><span class="secnum">RAW PRICE TAPE</span>'
    + '<h2>Last rows of the actual CSV files used to build features</h2></div><div class="card">'
    + table(['Date'].concat(Object.keys(series)),
      dates.map(function(d,i){
        return '<tr><td>' + esc(d) + '</td>' + Object.keys(series).map(function(k){
          const v = series[k][i];
          return '<td>' + (v===null||v===undefined?'n/a':num(v,2)) + '</td>';
        }).join('') + '</tr>';
      }))
    + '<p class="note" style="margin-top:10px">Level features use raw close; adjusted close '
    + 'is reserved for forward-outcome labels. Snapshot read directly from data/raw/*.csv.</p>'
    + '</div></section>';
  return h;
};

/* ---------------------------------------------------------------------------
 * OPERATIONS — daily refresh, percentiles, alerts, email.
 * ------------------------------------------------------------------------- */
RENDER.ops = function(){
  const O = D.ops || {};
  const st = O.status || {}, P = O.percentiles || {}, rules = O.rules || [];
  const alerts = O.recent_alerts || [], hist = O.score_history || [];
  const emcfg = O.email_config || {}, prev = O.email_preview || {};
  let h = '<section><div class="sechead"><span class="secnum">OPERATIONS</span>'
    + '<h2>Never miss a rare buying opportunity</h2>'
    + '<span class="sechnote">Daily refresh · score · rank · alert</span></div>'
    + '<div class="qgrid4">'
    + '<div class="qcard"><div class="qlabel">LAST DAILY RUN</div><div class="qans small">'
      + esc(st.last_run ? st.last_run.slice(0,16).replace('T',' ') : 'not run yet')
      + '</div><div class="qdet">' + esc(st.mode || 'offline') + ' · ' + esc(st.status || 'idle') + '</div></div>'
    + '<div class="qcard"><div class="qlabel">DATA REFRESH</div><div class="qans small">'
      + esc(st.refresh_status || 'not run') + '</div><div class="qdet">'
      + num(st.sources_updated,0) + ' sources updated · ' + num(st.sources_failed,0) + ' kept cached</div></div>'
    + '<div class="qcard"><div class="qlabel">VOO PERCENTILE</div><div class="qans">'
      + num(P.VOO,1) + '</div><div class="qdet">versus month-end scores since 2000</div></div>'
    + '<div class="qcard"><div class="qlabel">EMAIL DELIVERY</div><div class="qans small">'
      + esc(emcfg.mode || 'dry-run') + '</div><div class="qdet">'
      + (emcfg.configured ? 'SMTP configured' : 'Set SMTP env vars to enable sending') + '</div></div>'
    + '</div></section>';

  h += '<section><div class="sechead"><span class="secnum">ALERT RULES</span>'
    + '<h2>What triggers an email — and what never does</h2></div><div class="card">'
    + table(['Rule','Trigger','Current','State','Last fired'],
      rules.map(function(r){
        return '<tr><td>' + esc(r.name) + '</td><td class="lt">' + esc(r.trigger)
          + '</td><td>' + esc(r.current) + '</td><td>' + badge(r.state)
          + '</td><td>' + esc(r.last_fired || 'never') + '</td></tr>';
      }))
    + '<p class="note" style="margin-top:10px">Daily runs with no crossing, no regime change '
    + 'and no detector trigger send nothing. One weekly summary is the only recurring email. '
    + 'State is persisted so a repeat run cannot spam.</p></div></section>';

  h += '<section><div class="two"><div class="card"><h3>Score history (recent)</h3>'
    + table(['Date','VOO','QQQ','Regime','Detector'],
      hist.slice(-14).reverse().map(function(x){
        return '<tr><td>' + esc(x.date) + '</td><td>' + num(x.voo,1) + '</td><td>'
          + num(x.qqq,1) + '</td><td>' + esc(x.regime) + '</td><td>' + esc(x.detector) + '</td></tr>';
      }))
    + '</div><div class="card"><h3>Latest alert / email preview</h3>'
    + kv('Subject', prev.subject || 'no email generated yet')
    + kv('Archived alert', prev.latest_type || 'n/a')
    + kv('Date', prev.date || D.meta.as_of)
    + kv('VOO / QQQ', prev.scores || (num(D.instruments.VOO.score,1) + ' / ' + num(D.instruments.QQQ.score,1)))
    + kv('Percentile', prev.percentile || (num(P.VOO,1) + ' / ' + num(P.QQQ,1)))
    + kv('Regime', prev.regime || D.hero.market_regime)
    + kv('Deployment', prev.deployment || D.hero.deployment)
    + '<p class="note" style="margin-top:10px">Preview of the most recent email archived '
    + 'in data/ops/outbox/. Full HTML, text and JSON copies are kept there; delivery is '
    + 'dry-run unless SMTP credentials are set.</p>'
    + '<p class="note" style="margin-top:8px">' + esc(O.weekly_summary && O.weekly_summary.text
        ? O.weekly_summary.text : 'Weekly summary is generated on the first run of each ISO week.')
    + '</p></div></div></section>';

  h += '<section><div class="sechead"><span class="secnum">ALERT LOG</span>'
    + '<h2>Emails evaluated</h2></div><div class="card">'
    + (alerts.length
      ? table(['Time','Type','Priority','Delivery','Detail'], alerts.slice().reverse().map(function(a){
          return '<tr><td>' + esc(a.time ? a.time.slice(0,16).replace('T',' ') : '') + '</td><td>'
            + esc(a.type) + '</td><td>' + esc(a.priority || 'normal') + '</td><td>'
            + esc(a.delivery || 'dry-run') + '</td><td class="lt">' + esc(a.detail) + '</td></tr>';
        }))
      : '<p class="note">No alerts yet. That is expected: today does not cross a threshold, '
        + 'the regime has not changed and the rare detector is not triggered.</p>')
    + '</div></section>';

  h += '<section><div class="sechead"><span class="secnum">SETUP</span>'
    + '<h2>How the daily run is wired</h2></div><div class="card">'
    + '<div class="formula">' + esc(O.setup && O.setup.command
        ? O.setup.command
        : 'python pipeline/ops_daily.py --no-refresh --dry-run') + '</div>'
    + '<ul class="clean" style="margin-top:12px">'
    + (O.setup && O.setup.notes ? O.setup.notes.map(function(n){ return '<li>' + esc(n) + '</li>'; }).join('')
      : '<li>Run the pipeline daily; it refreshes data, recomputes the score, ranks it against history and evaluates alerts.</li>')
    + '</ul></div></section>';
  return h;
};

RENDER.analogues = function(){
  let h = '<section><div class="sechead"><span class="secnum">HISTORICAL ANALOGUES</span>'
    + '<h2>Today versus five defining selloffs</h2></div><div class="qgrid3">';
  D.analogues.forEach(function(a){
    h += '<div class="simcard"><div class="simtop"><span class="simname">' + esc(a.episode)
      + '</span><span class="simscore">' + num(a.similarity,1) + '%</span></div>'
      + '<div class="note">' + esc(a.date) + '</div>'
      + '<div class="simbar"><i style="width:' + a.similarity + '%;background:'
      + colorFor(a.similarity) + '"></i></div>'
      + '<div class="simfeat">VIX ' + num(a.features.vix,1) + ' · drawdown '
      + num(a.features.dd*100,1) + '% · &gt;200DMA ' + num(a.features.above,0) + '%</div></div>';
  });
  h += '</div></section><section><div class="sechead"><span class="secnum">VALIDATION</span>'
    + '<h2>Panic detector on historical stress episodes</h2></div><div class="card">'
    + '<div class="scroll"><table><thead><tr><th>Date</th><th>Conditions</th><th>Detector</th></tr></thead><tbody>'
    + D.historical_validation.map(function(x){ return '<tr><td>' + esc(x.date) + '</td><td>'
      + x.n_conditions + '/5</td><td>' + badge(x.detector) + '</td></tr>'; }).join('')
    + '</tbody></table></div></div></section>';
  return h;
};

RENDER.method = function(){
  let h = '<section><div class="sechead"><span class="secnum">METHOD</span><h2>Design and limits</h2></div>'
    + '<div class="card"><p class="note">' + esc(D.meta.philosophy) + ' Inherited: '
    + esc(D.meta.inherited) + '.</p><h3>Disclosures</h3><ul class="clean">'
    + D.disclosures.map(function(x){ return '<li>' + esc(x) + '</li>'; }).join('')
    + '</ul></div></section>';
  return h;
};

RENDER.footer = function(){
  return '<div>' + esc(D.meta.version) + ' · generated ' + esc(D.meta.generated_at) + '</div>'
    + '<div style="margin-top:5px">Observable conditions are not return predictions. '
    + 'Past episodes do not guarantee future results. Not investment advice.</div>';
};

function setTab(name){
  if (!TABS.some(function(t){ return t[0] === name; })) name = 'decision';
  activeTab = name;
  TABS.forEach(function(t){
    const btn = document.getElementById('tab-' + t[0]);
    const pane = document.getElementById('pane-' + t[0]);
    if (btn && btn.className !== undefined) btn.className = (t[0]===name?'active':'');
    if (pane && pane.className !== undefined) pane.className = 'tabpane' + (t[0]===name?' active':'');
  });
  try {
    if (window.location && window.location.hash !== undefined) window.location.hash = name;
  } catch (e) { /* file:// or sandboxed context */ }
}
function setMode(mode){
  expert = (mode === 'expert');
  if (document.body && document.body.className !== undefined) document.body.className = expert ? 'expert' : '';
  RENDER.refreshWhy();
}
RENDER.refreshWhy = function(){
  const pane = document.getElementById('pane-why');
  if (pane) pane.innerHTML = RENDER.why();
};

function handleClick(e){
  const t = e && e.target;
  if (!t || !t.getAttribute) return;
  const tab = t.getAttribute('data-tab');
  if (tab) { setTab(tab); return; }
  const mode = t.getAttribute('data-mode');
  if (mode) setMode(mode);
}

function boot(){
  document.getElementById('metastrip').innerHTML = RENDER.meta();
  document.getElementById('hero').innerHTML = RENDER.hero();
  document.getElementById('tabbar').innerHTML = RENDER.tabs();
  document.getElementById('pane-decision').innerHTML = RENDER.decision();
  document.getElementById('pane-regime').innerHTML = RENDER.regime();
  document.getElementById('pane-credit').innerHTML = RENDER.credit();
  document.getElementById('pane-internals').innerHTML = RENDER.internals();
  document.getElementById('pane-cost').innerHTML = RENDER.cost();
  RENDER.refreshWhy();
  document.getElementById('pane-data').innerHTML = RENDER.data();
  document.getElementById('pane-analogues').innerHTML = RENDER.analogues();
  document.getElementById('pane-ops').innerHTML = RENDER.ops();
  document.getElementById('pane-method').innerHTML = RENDER.method();
  document.getElementById('sitefooter').innerHTML = RENDER.footer();

  /* Delegated listeners live on persistent containers, so rebuilding a pane
     never destroys navigation. Direct button listeners are kept as a fallback. */
  const bar = document.getElementById('tabbar');
  if (bar && bar.addEventListener) bar.addEventListener('click', handleClick);
  const whyPane = document.getElementById('pane-why');
  if (whyPane && whyPane.addEventListener) whyPane.addEventListener('click', handleClick);
  if (document.addEventListener) document.addEventListener('click', handleClick);
  TABS.forEach(function(t){
    const btn = document.getElementById('tab-' + t[0]);
    if (btn && btn.addEventListener) btn.addEventListener('click', function(){ setTab(t[0]); });
  });

  /* Hash deep-link: index_v7.html#data opens the Data tab directly. */
  try {
    const hash = window.location && window.location.hash ? window.location.hash.replace('#','') : '';
    if (hash) setTab(hash);
  } catch (e) { /* ignore */ }
}
boot();
