/* =====================================================================
   ETF Timing Dashboard — v2 validation harness
   ---------------------------------------------------------------------
   Loads the FINAL built HTML, stubs the DOM/ECharts, and runs the real
   application code (including init()) with a controllable mock fetch.

   Covers:
     1. live fetch success (all six symbols, correct URLs, symbol verification)
     2. embedded fallback
     3. partial fallback (live + embedded mix)
     4. stale-data detection (lag + feed)
     5. source attribution / confidence scoring
     6. score reproducibility (embedded and live paths)
     7. trend calculation consistency (MA, momentum, heatmap, interpretation)
     8. retry with exponential backoff, symbol-mismatch rejection,
        no fabricated data for out-of-range dates

   Run:  node test_dashboard_v2.js          (offline, deterministic mocks)
         node test_dashboard_v2.js --live   (also hits Yahoo for a smoke test)
   ===================================================================== */
const fs = require('fs');

const HTML = fs.readFileSync('ETF_Timing_Dashboard.html', 'utf8');
const PAYLOAD = fs.readFileSync('dashboard_data.json', 'utf8');
const D = JSON.parse(PAYLOAD);

/* ---------- minimal DOM / browser stubs ---------- */
function makeEl(id) {
  return {
    id, innerHTML: '', textContent: '', value: '', disabled: false,
    min: '', max: '', checked: false,
    style: {}, dataset: {}, children: [],
    classList: { add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(){}, getAttribute(){ return null; },
    addEventListener(){}, appendChild(){}, removeChild(){},
    querySelector(){ return makeEl('q'); }, querySelectorAll(){ return []; },
    focus(){}, scrollIntoView(){}, getBoundingClientRect(){ return {width:600,height:340}; }
  };
}
const els = new Map();
function getEl(id) { if(!els.has(id)) els.set(id, makeEl(id)); return els.get(id); }
// Seed values that a real browser would derive from HTML attributes / <option>s.
['staleLagDays','staleFeedDays'].forEach(k=>{});
getEl('staleLagDays').value = '5';
getEl('staleFeedDays').value = '7';
getEl('sigSelect').value = '0';
getEl('inMonth').value = '11';
getEl('cmpA').value = '2008-09-17';
getEl('cmpB').value = D.dates[D.n-1];

const documentStub = {
  getElementById: getEl,
  querySelectorAll(){ return []; },
  querySelector(){ return makeEl('q'); },
  documentElement: { setAttribute(){}, getAttribute(){ return 'light'; }, style:{} },
  body: makeEl('body'),
  addEventListener(){}
};
const windowStub = { addEventListener(){}, scrollTo(){}, localStorage:null };
const storageMap = new Map();
// Simulate a configured Cloudflare Worker for the main offline suite.
const TEST_WORKER_BASE = 'https://worker.example.com';
storageMap.set('etfWorkerBase', TEST_WORKER_BASE);
const localStorageStub = {
  getItem(k){ return storageMap.has(k)?storageMap.get(k):null; },
  setItem(k,v){ storageMap.set(k,String(v)); },
  removeItem(k){ storageMap.delete(k); }
};
const getComputedStyleStub = () => ({ getPropertyValue(){ return '#3b82f6'; } });
const chartStub = { lastOption:null, setOption(o){ this.lastOption=o; }, resize(){}, dispose(){}, on(){} };
const echartsStub = { init(){ return chartStub; } };

/* ---------- mutable network mock ---------- */
let currentFetch = async () => { throw new Error('no mock configured'); };
const fetchProxy = (...args) => currentFetch(...args);

/* ---------- extract the inline application script ---------- */
const scripts = [...HTML.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]);
let code = scripts[scripts.length-1];
code = code.split('\n').filter(l => !/^const D = \{/.test(l) && !/^const N = D\.n;/.test(l)).join('\n');

const EXPORTS = [
  'DATA_CONFIG','SYMBOL_KEYS','VERIFY_LINKS','buildQuoteUrls','buildQuoteUrl',
  'normaliseSymbol','daysBetween','parseHistoryJSON','fetchMarketHistory',
  'loadSeriesForDate','getEmbeddedSeries','embeddedCoversDate','embeddedLoadedFor',
  'calculateIndicators','validateIndicators','indicatorsToScoreInput','computeScore',
  'detectSignals','assessDataQuality','buildAuditRows','buildManualAuditRows',
  'movingAverage','scoreTrendStats','scoreHeatmapData','bandOf','thresholds',
  'generateScore','generateLatestScore','idxOfDate','compsFromArrays',
  'DIAG','DIAG_STORE','diagPush','classifyFetchError','friendlyError','keyForSymbol',
  'renderDebugPanel','renderRawViewer','renderValidation','renderHealthPanel',
  'renderAuditLogPanel','renderTransparency','renderDiagnosticPanels',
  'testLiveConnection','clearDiagnostics','clearAuditLog','setHtml','getChart',
  'appendAuditLog','recordHealth','storeRaw',
  'RUNTIME','isFileMode','friendlyUnavailableHtml','setEmbeddedOnly',
  'continueWithEmbedded','exitEmbeddedMode','initRuntimeMode',
  'DEPLOY_CONFIG','PROVIDER_STATUS','getWorkerBase',
  'probeProviderStatus','renderProviderStatusPanel','renderHistoryChart',
  'renderTrendScoreChart','ONE_YEAR_SESSIONS','currentProviderLabel'
];
const locationStub = { protocol: 'http:' };
const factory = new Function(
  'D','document','window','localStorage','getComputedStyle','echarts','fetch','location',
  'const N = D.n;\n' + code + '\nreturn {' + EXPORTS.join(',') + ', getLastResult:()=>lastResult, getAuditRows:()=>auditRowsCache, getTrendMarker:()=>trendMarkerDate };'
);
const api = factory(D, documentStub, windowStub, localStorageStub,
                    getComputedStyleStub, echartsStub, fetchProxy, locationStub);
// keep tests fast: retry backoff is exercised explicitly in a dedicated test
api.DATA_CONFIG.retries = 0;
api.DATA_CONFIG.backoffBaseMs = 1;

/* ---------- test utilities ---------- */
let pass=0, fail=0;
const results=[];
function check(cond,msg){ if(cond) pass++; else { fail++; results.push('FAIL: '+msg); console.log('  ✗ '+msg); } }
function section(t){ console.log('\n=== '+t+' ==='); }

/** Build a Yahoo-shaped response for a raw symbol, from embedded rows. */
function yahooShape(symbol, rows){
  return { chart:{ error:null, result:[{
    meta:{ symbol, currency:'USD', exchangeName:'TEST' },
    timestamp: rows.map(r=>Math.floor(Date.parse(r.date+'T00:00:00Z')/1000)),
    indicators:{ quote:[{ close: rows.map(r=>r.close) }] }
  }] } };
}
/** Live-shaped rows sourced from the embedded dataset (up to a date). */
function liveRowsUpTo(k, date, dropLast=0){
  const out=[];
  for(let i=0;i<D.n;i++){
    if(D.dates[i]>date) break;
    const v=D[k][i];
    if(typeof v==='number' && isFinite(v)) out.push({date:D.dates[i],close:v});
  }
  return dropLast?out.slice(0,-dropLast):out;
}
function allLiveUpTo(date, dropLast=0){
  const o={}; api.SYMBOL_KEYS.forEach(k=>o[k]=liveRowsUpTo(k,date,dropLast)); return o;
}
/** Configure the mock: set of symbols that succeed, and optional row overrides. */
function mockSucceed(successSet, date, opts={}){
  const rows=opts.rows||allLiveUpTo(date,opts.dropLast||0);
  // map RAW provider symbols ('^VIX') -> internal keys ('VIX')
  const byRaw={}; api.SYMBOL_KEYS.forEach(k=>{ byRaw[api.DATA_CONFIG.symbols[k]]=rows[k]||[]; });
  currentFetch=async(url)=>{
    const u=String(url);
    const y=/\/chart\/([^?]+)\?/.exec(u);
    const w=/[?&]symbol=([^&]+)/.exec(u);
    const sym=y?decodeURIComponent(y[1]):(w?decodeURIComponent(w[1]):'');
    if(!successSet.has(sym)) throw new Error('simulated provider failure for '+sym);
    const body=yahooShape(sym,byRaw[sym]||[]);
    return {ok:true,status:200,json:async()=>opts.transform?opts.transform(sym,body):body};
  };
}
function mockAllFail(){ currentFetch=async()=>{ const e=new TypeError('Failed to fetch'); throw e; }; }
const OK_DATE='2020-03-23';
const OK_IDX=D.dates.indexOf(OK_DATE);

/* ===================================================================== */
(async()=>{
console.log('ETF Timing Dashboard — v2 validation');
console.log('embedded rows: '+D.n+'  |  range: '+D.dates[0]+' → '+D.dates[D.n-1]);
console.log('init() ran with DOM stubs: '+(api.getTrendMarker()!==null?'OK':'FAILED'));
check(api.getTrendMarker()!==null,'init() completed and trend section initialised');
check(api.getAuditRows().length===8,'embedded preview populates 8 audit rows');

section('1. Symbol handling & URL encoding (double-encoding regression)');
const p1=1700000000,p2=1763000000;
api.SYMBOL_KEYS.forEach(k=>{
  const sym=api.DATA_CONFIG.symbols[k];
  const urls=api.buildQuoteUrls(sym,p1,p2);
  const u=urls[0].url;
  check(!u.includes('%25'),k+' URL is not double-encoded: '+u);
  check(u.includes(encodeURIComponent(sym)),k+' URL contains exactly-once encoding '+encodeURIComponent(sym));
  check(urls[0].name==='worker'&&u.startsWith(TEST_WORKER_BASE+'/chart?symbol='),k+' Worker is the first provider');
  console.log('  '+k.padEnd(4)+' raw='+sym.padEnd(5)+' → '+encodeURIComponent(sym));
});
check(api.DATA_CONFIG.directYahooEnabled===false,'direct browser-to-Yahoo fetch disabled by default');
check(api.getWorkerBase()===TEST_WORKER_BASE,'worker base resolved from browser setting');
check(api.DATA_CONFIG.symbols.VIX==='^VIX','VIX stored as raw caret symbol');
check(api.DATA_CONFIG.symbols.TNX==='^TNX','TNX stored as raw caret symbol');
check(api.DATA_CONFIG.symbols.IRX==='^IRX','IRX stored as raw caret symbol');
check(api.normaliseSymbol('%5EVIX')===api.normaliseSymbol('^VIX'),'normaliseSymbol treats encoded/raw caret as equal');

section('2. Provider JSON validation & symbol verification');
const good=yahooShape('^VIX',[{date:'2024-01-02',close:15},{date:'2024-01-03',close:16}]);
const parsed=api.parseHistoryJSON(good,'^VIX','test');
check(parsed.rows.length===2,'Yahoo chart shape parsed');
check(parsed.symbolVerified===true&&parsed.reportedSymbol==='^VIX','reported symbol verified against request');
let threw=false; try{ api.parseHistoryJSON(good,'SPY','test'); }catch(e){ threw=/Symbol mismatch/.test(e.message); }
check(threw,'wrong meta.symbol is rejected');
threw=false; try{ api.parseHistoryJSON({chart:{error:{code:'Not Found',description:'bad symbol'}}},'^VIX','t'); }catch(e){ threw=/provider error/.test(e.message); }
check(threw,'provider error payload rejected');
threw=false; try{ api.parseHistoryJSON({bad:1},'^VIX','t'); }catch(e){ threw=true; }
check(threw,'unrecognised JSON shape rejected');
const noMeta={chart:{result:[{timestamp:[1700000000],indicators:{quote:[{close:[10]}]}}]}};
threw=false; try{ api.parseHistoryJSON(noMeta,'^VIX','t'); }catch(e){ threw=/missing meta.symbol/.test(e.message); }
check(threw,'Yahoo response without meta.symbol rejected');
const proxyShape=api.parseHistoryJSON([{date:'2024-01-02',close:10}],'^VIX','t');
check(proxyShape.rows.length===1&&proxyShape.symbolVerified===false,'generic proxy shape accepted but flagged unverified');
const workerShape={ok:true,provider:'Cloudflare Worker → Yahoo Finance',symbol:'SPY',requestedSymbol:'SPY',
  data:[{date:'2024-01-03',close:470},{date:'2024-01-02',close:472}],cached:true};
const wp=api.parseHistoryJSON(workerShape,'SPY','worker-url');
check(wp.rows.length===2&&wp.rows[0].date==='2024-01-02','worker payload parsed and date-sorted');
check(wp.symbolVerified===true&&wp.provider.includes('Cloudflare Worker')&&wp.cached===true,'worker symbol/provider/cache metadata preserved');
let workerErr=false; try{ api.parseHistoryJSON({ok:false,error:{code:'UPSTREAM_HTTP_429',status:429,message:'rate limit'}},'SPY','w'); }catch(e){ workerErr=/HTTP 429/.test(e.message)&&/UPSTREAM_HTTP_429/.test(e.message); }
check(workerErr,'worker structured error surfaced with HTTP status and code');

section('3. Live fetch success — all six symbols');
mockSucceed(new Set(['SPY','QQQ','^VIX','TLT','^TNX','^IRX']), OK_DATE);
const live=await api.loadSeriesForDate(OK_DATE);
console.log('  source='+live.source+' live='+live.liveCount+' embedded='+live.embeddedCount+' missing='+live.missingCount);
check(live.source==='live','all-live source');
check(live.liveCount===6&&live.embeddedCount===0&&live.missingCount===0,'six live / zero fallback');
api.SYMBOL_KEYS.forEach(k=>{
  const p=live.provenance[k];
  check(p.source==='live'&&p.provider.includes('Cloudflare Worker'),k+' attributed to Cloudflare Worker');
  check(p.providerKind==='worker'&&p.url.startsWith(TEST_WORKER_BASE+'/chart?'),k+' records worker kind and exact URL');
  check(p.symbolVerified===true&&p.reportedSymbol===api.DATA_CONFIG.symbols[k],k+' response symbol verified');
  check(p.retrievedAt instanceof Date,k+' has retrieval timestamp');
  check(!!p.verifyUrl&&!!p.url,k+' has verification + request URL');
});
const qLive=api.assessDataQuality(live,OK_DATE,OK_DATE,5,7);
check(qLive.confidence===100,'100% live-data confidence');
check(qLive.liveCount===6&&qLive.embeddedCount===0,'quality counts match');
check(live.currentProvider==='worker'&&live.providerCounts.worker===6,'current provider is Worker with 6 symbols');
const liveInd=api.calculateIndicators(live.series,OK_DATE);
const liveScore=api.computeScore(api.indicatorsToScoreInput(liveInd.ind));
check(Math.abs(liveScore.total-D.opp_score[OK_IDX])<0.01,'live-path score reproduces stored score ('+liveScore.total+')');

section('4. Embedded fallback (total live failure)');
mockAllFail();
const emb=await api.loadSeriesForDate(OK_DATE);
console.log('  source='+emb.source+' live='+emb.liveCount+' embedded='+emb.embeddedCount);
check(emb.source==='embedded','falls back to embedded');
check(emb.liveCount===0&&emb.embeddedCount===6,'all six substituted from embedded');
const qEmb=api.assessDataQuality(emb,OK_DATE,OK_DATE,5,7);
check(qEmb.confidence===0,'0% confidence when nothing is live');
check(qEmb.warnings.some(w=>/Fed Funds/.test(w)),'Fed Funds unavailability is disclosed');
check(qEmb.warnings.some(w=>/embedded historical data/.test(w)),'embedded substitution is warned');
const embInd=api.calculateIndicators(emb.series,OK_DATE);
check(Math.abs(api.computeScore(api.indicatorsToScoreInput(embInd.ind)).total-D.opp_score[OK_IDX])<0.01,'embedded-path score reproduces stored score');

section('5. Partial fallback — live where possible, embedded only for failures');
mockSucceed(new Set(['SPY','QQQ','^VIX','TLT']), OK_DATE);  // ^TNX and ^IRX fail
const part=await api.loadSeriesForDate(OK_DATE);
console.log('  source='+part.source+' live='+part.liveCount+' embedded='+part.embeddedCount);
check(part.source==='partial','partial source classification');
check(part.liveCount===4&&part.embeddedCount===2,'4 live + 2 embedded');
['SPY','QQQ','VIX','TLT'].forEach(k=>check(part.provenance[k].source==='live',k+' stayed live'));
['TNX','IRX'].forEach(k=>check(part.provenance[k].source==='embedded',k+' substituted from embedded'));
const qPart=api.assessDataQuality(part,OK_DATE,OK_DATE,5,7);
check(qPart.confidence===67,'confidence = round(4/6) = 67');
check(qPart.warnings.some(w=>/Partial live retrieval/.test(w)),'partial-mix warning raised');
const partInd=api.calculateIndicators(part.series,OK_DATE);
check(Math.abs(api.computeScore(api.indicatorsToScoreInput(partInd.ind)).total-D.opp_score[OK_IDX])<0.01,'mixed-source score still reproducible');

section('6. Stale-data detection');
mockSucceed(new Set(['SPY','QQQ','^VIX','TLT','^TNX','^IRX']), OK_DATE, {dropLast:20});
const stale=await api.loadSeriesForDate(OK_DATE);
const staleInd=api.calculateIndicators(stale.series,OK_DATE);
const qStale=api.assessDataQuality(stale,OK_DATE,staleInd.marketDate,5,7);
console.log('  lag warnings: '+qStale.warnings.filter(w=>/calendar days older/.test(w)).length);
check(qStale.warnings.some(w=>/calendar days older than the requested date/.test(w)),'stale lag warning triggered above threshold');
check(Object.values(qStale.rows).some(r=>r.staleLag===true),'per-metric staleLag flag set');
// Same data at a tighter/looser threshold
const qLoose=api.assessDataQuality(stale,OK_DATE,staleInd.marketDate,60,7);
check(!qLoose.warnings.some(w=>/calendar days older/.test(w)),'configurable threshold suppresses warning at 60 days');
// Feed staleness only for "latest" requests
const today=new Date().toISOString().slice(0,10);
mockSucceed(new Set(['SPY','QQQ','^VIX','TLT','^TNX','^IRX']), today,
  {rows:allLiveUpTo(D.dates[D.n-1],12)});
const feed=await api.loadSeriesForDate(today);
const feedInd=api.calculateIndicators(feed.series,today);
const qFeed=api.assessDataQuality(feed,today,feedInd.marketDate,5,7);
check(qFeed.warnings.some(w=>/provider feed may be delayed/.test(w)),'stale feed warning triggered for latest request');
const qHist=api.assessDataQuality(live,OK_DATE,OK_DATE,5,7);
check(!qHist.warnings.some(w=>/provider feed may be delayed/.test(w)),'historical request is NOT falsely flagged as stale feed');

section('7. Source attribution & audit panel rows');
const auditRows=api.buildAuditRows(qPart,partInd.ind,OK_DATE);
check(auditRows.length===8,'8 audit metrics (6 raw + Fed Funds + Yield Curve)');
check(auditRows.map(r=>r.metric).includes('Fed Funds'),'Fed Funds row present');
check(auditRows.find(r=>r.metric==='Fed Funds').value==='Not available','Fed Funds NOT fabricated');
check(auditRows.find(r=>r.metric==='Fed Funds').verify.includes('FEDFUNDS'),'Fed Funds FRED verification link');
check(auditRows.find(r=>r.metric.startsWith('Yield Curve')).source==='derived','Yield Curve marked Derived');
check(auditRows.find(r=>r.metric.startsWith('Yield Curve')).verify.includes('T10Y2Y'),'Yield Curve FRED cross-check link');
check(api.VERIFY_LINKS.length===8,'8 clickable verification links');
['SPY','QQQ','%5EVIX','%5ETNX','%5EIRX','TLT'].forEach(s=>check(api.VERIFY_LINKS.some(l=>l.url.includes(s)),'verify link for '+s));
check(api.VERIFY_LINKS.some(l=>l.url.includes('FRED')===false&&l.name==='FEDFUNDS') || api.VERIFY_LINKS.some(l=>l.name==='FEDFUNDS'),'FRED FEDFUNDS link');
check(api.VERIFY_LINKS.some(l=>l.name==='T10Y2Y'),'FRED T10Y2Y link');
const manualRows=api.buildManualAuditRows();
check(manualRows.filter(r=>r.source==='manual').length===5,'manual rows attributed as Manual');
check(manualRows.every(r=>r.retrievedAt===null||r.source!=='live'),'manual rows never reported as live');

section('8. Retry with exponential backoff');
api.DATA_CONFIG.retries=2; api.DATA_CONFIG.backoffBaseMs=1;
  let calls=0;
  const rows=allLiveUpTo(OK_DATE);
  const byRaw={}; api.SYMBOL_KEYS.forEach(k=>{ byRaw[api.DATA_CONFIG.symbols[k]]=rows[k]||[]; });
  currentFetch=async(url)=>{
    calls++;
    const u=String(url);
    const y=/\/chart\/([^?]+)\?/.exec(u), w=/[?&]symbol=([^&]+)/.exec(u);
    const sym=y?decodeURIComponent(y[1]):decodeURIComponent(w[1]);
    if(calls<=2) throw new Error('transient');
    return {ok:true,status:200,json:async()=>yahooShape(sym,byRaw[sym]||[])};
  };
const t0=Date.now();
const retried=await api.fetchMarketHistory('^VIX','2019-01-01',OK_DATE);
check(retried.rows.length>0,'fetch succeeds after transient failures');
check(retried.attempts===3,'3 attempts recorded (2 retries) — got '+retried.attempts);
check(Date.now()-t0>=2,'backoff delays applied');
api.DATA_CONFIG.retries=0;

section('9. No fabricated data / no look-ahead');
mockAllFail();
const future=await api.loadSeriesForDate('2030-01-01');
console.log('  future-date source='+future.source+' missing='+future.missingCount);
check(future.source==='error','out-of-range date returns error, not a stale score');
check(future.missingCount===6,'all six symbols marked missing for out-of-range date');
check(future.error.includes('outside embedded coverage'),'error explains why fallback was refused');
// No look-ahead: recompute across the whole history from truncated raw series
let mismatches=0, maxDiff=0;
const startI=D.dates.indexOf('2003-04-01');
for(let i=startI;i<D.n;i+=53){
  const dt=D.dates[i], s={};
  api.SYMBOL_KEYS.forEach(k=>{ s[k]=liveRowsUpTo(k,dt); });
  const r=api.calculateIndicators(s,dt);
  const sc=api.computeScore(api.indicatorsToScoreInput(r.ind));
  const diff=Math.abs(sc.total-D.opp_score[i]);
  maxDiff=Math.max(maxDiff,diff);
  if(diff>0.01) mismatches++;
  check(r.marketDate===dt,'trading date '+dt);
  check(sc.total===sc.trend+sc.dip+sc.vix+sc.macro+sc.season,'component sum '+dt);
  check(sc.total>=0&&sc.total<=100,'score range '+dt);
}
console.log('  sampled '+Math.ceil((D.n-startI)/53)+' dates, max diff='+maxDiff.toFixed(3)+', mismatches='+mismatches);
check(mismatches===0,'score_history reproducibility across full history');

section('10. Key-date integrity & date resolution');
[['2024-06-14','2024-06-14'],['2026-11-15','2026-11-13'],['2024-12-25','2024-12-24'],
 ['2008-09-17','2008-09-17'],['2020-03-23','2020-03-23'],['2022-10-12','2022-10-12']].forEach(([req,exp])=>{
  const s={}; api.SYMBOL_KEYS.forEach(k=>s[k]=liveRowsUpTo(k,req));
  const r=api.calculateIndicators(s,req);
  const sc=api.computeScore(api.indicatorsToScoreInput(r.ind));
  check(r.marketDate===exp,'date resolution '+req+' → '+exp);
  console.log('  '+req+' → '+r.marketDate+'  score='+sc.total+' ['+api.bandOf(sc.total)+']');
});

section('11. Buy Score Trend analytics consistency');
const ti=D.n-1;
const st=api.scoreTrendStats(ti);
// independent recomputation
function manualMA(arr,w,i){ let s=0,c=0; for(let j=Math.max(0,i-w+1);j<=i;j++){s+=arr[j];c++;} return s/c; }
check(Math.abs(st.ma30-manualMA(D.opp_score,30,ti))<1e-9,'MA30 matches independent computation');
check(Math.abs(st.ma90-manualMA(D.opp_score,90,ti))<1e-9,'MA90 matches independent computation');
check(Math.abs(st.d30-(D.opp_score[ti]-D.opp_score[ti-30]))<1e-9,'30-session change matches');
check(Math.abs(st.d90-(D.opp_score[ti]-D.opp_score[ti-90]))<1e-9,'90-session change matches');
check(Math.abs(st.ma30Prev-manualMA(D.opp_score,30,ti-30))<1e-9,'MA30 slope baseline uses only past data');
check(st.votes.length===4&&st.net===st.votes.reduce((a,b)=>a+b,0),'4 votes sum to net');
check(['Improving','Stable','Weakening'].includes(st.interpretation),'interpretation is one of three labels');
console.log('  '+st.date+': score='+st.current+' MA30='+st.ma30.toFixed(2)+' MA90='+st.ma90.toFixed(2)+
            ' d30='+st.d30.toFixed(1)+' d90='+st.d90.toFixed(1)+' net='+st.net+' → '+st.interpretation);
// determinism + monotonic vote rule
check(api.scoreTrendStats(ti).interpretation===st.interpretation,'interpretation deterministic');
const forced=api.scoreTrendStats(500);
check(['Improving','Stable','Weakening'].includes(forced.interpretation),'interpretation valid at mid-history index');
check(api.movingAverage([1,2,3,4],2,3)===3.5,'movingAverage edge behaviour');
check(api.movingAverage([],30,0)===null,'movingAverage returns null for empty input');
const hm=api.scoreHeatmapData(ti);
check(hm.months.length===12&&hm.weeks.length===5,'heatmap is 12 months × 5 week buckets');
check(hm.cells.length===60,'60 heatmap cells');
check(hm.cells.every(c=>c[2]===null||(c[2]>=0&&c[2]<=100)),'all heatmap scores in 0–100');
check(hm.cells.some(c=>c[2]!==null),'heatmap has real observations');
console.log('  heatmap: '+hm.months[0].label+' → '+hm.months[11].label+', non-empty cells='+hm.cells.filter(c=>c[2]!==null).length);

section('12. Full high-level workflow through generateScore()');
mockSucceed(new Set(['SPY','QQQ','^VIX','TLT','^TNX','^IRX']), OK_DATE);
getEl('datePick').value=OK_DATE;
await api.generateScore();
const lr=api.getLastResult();
check(!!lr&&lr.source==='live','generateScore produced a live result');
check(!!lr.quality&&lr.quality.confidence===100,'generateScore attached 100% quality assessment');
check(api.getAuditRows().length===8&&api.getAuditRows()[0].source==='live','audit panel refreshed with live rows');
check(!!getEl('qualityKpis').innerHTML&&getEl('qualityKpis').innerHTML.includes('100%'),'quality KPI rendered');
check(!!getEl('trendInterp').innerHTML,'trend interpretation rendered');
console.log('  generateScore → '+lr.marketDate+' score='+lr.score.total+' source='+lr.source+' confidence='+lr.quality.confidence+'%');
mockAllFail();
await api.generateScore();
const lr2=api.getLastResult();
check(lr2.source==='embedded','generateScore falls back to embedded on network failure');
check(lr2.quality.confidence===0,'fallback confidence 0%');
mockAllFail();
await api.generateLatestScore();
check(!!api.getLastResult(),'generateLatestScore completes even when live data fails');
console.log('  generateLatestScore → source='+api.getLastResult().source+' marketDate='+api.getLastResult().marketDate);

/* ---------- optional real-network smoke test ---------- */
/* ---------- 13–20: transparency / diagnostics ---------- */
section('13. Failure classification and user-readable messages');
check(api.classifyFetchError(new Error('HTTP 429')).code==='HTTP_429','HTTP 429 classified');
check(/HTTP 429/.test(api.friendlyError(new Error('HTTP 429'),'Fetch data')),'HTTP 429 message is specific');
check(api.classifyFetchError(new TypeError('Failed to fetch')).code==='CORS_BLOCKED','CORS classified');
check(/live provider could not be reached/.test(api.friendlyError(new TypeError('Failed to fetch'),'Fetch data')),'network failure message is specific to the Worker architecture');
const abortErr=new Error('aborted'); abortErr.name='AbortError';
check(api.classifyFetchError(abortErr).code==='TIMEOUT','abort classified as timeout');
check(/Timeout after 9000ms/.test(api.friendlyError(abortErr,'Fetch data')),'timeout message states configured limit');
check(api.classifyFetchError(new Error('invalid JSON: Unexpected token')).code==='INVALID_JSON','invalid JSON classified');
check(/Response JSON invalid/.test(api.friendlyError(new Error('invalid JSON: x'),'Fetch data')),'invalid JSON message is specific');
check(/Indicator calculation failed/.test(api.friendlyError(new Error('SPY: need at least 252'),'Calculate indicators')),'calculation phase message');
check(/Rendering failure/.test(api.friendlyError(new Error('x'),'Render dashboard')),'render phase message');
check(!/Could not generate score/.test(api.friendlyError(new Error('HTTP 429'),'Fetch data')),'generic message removed');

section('14. Live Connection Tester — all succeed');
const TEST_DATE=D.dates[D.n-1], ALL_ROWS=allLiveUpTo(TEST_DATE);
const ALL_RAW=new Set(api.SYMBOL_KEYS.map(k=>api.DATA_CONFIG.symbols[k]));
mockSucceed(ALL_RAW,TEST_DATE,{rows:ALL_ROWS});
api.clearDiagnostics(); api.clearAuditLog();
await api.testLiveConnection();
let grid=getEl('testLiveGrid').innerHTML;
check((grid.match(/✅/g)||[]).length===6,'six green ticks shown');
check((grid.match(/HTTP 200/g)||[]).length===6,'per-symbol HTTP status displayed');
check(/Live Data Available/.test(getEl('testLiveStatus').innerHTML),'overall status Live Data Available');
check(api.DIAG.log[0]&&api.DIAG.log[0].event==='Connection Test','connection test written to audit log');
check(api.DIAG.log[0].confidence===100,'connection test confidence 100%');

section('15. Live Connection Tester — partial and total failure');
mockSucceed(new Set(['SPY','QQQ','^VIX','TLT']),TEST_DATE,{rows:ALL_ROWS});
api.clearDiagnostics();
await api.testLiveConnection();
grid=getEl('testLiveGrid').innerHTML;
check(/Partial Retrieval/.test(getEl('testLiveStatus').innerHTML),'partial retrieval status');
check((grid.match(/❌/g)||[]).length===2,'two failed symbols displayed');
mockAllFail();
await api.testLiveConnection();
check(/Live Data Unavailable/.test(getEl('testLiveStatus').innerHTML),'all-fail status');
check(/Live market quotes are currently unavailable/.test(getEl('testLiveStatus').innerHTML),'friendly unavailable message shown in diagnostics');
check(/No action is required/.test(getEl('testLiveStatus').innerHTML),'no-action reassurance shown');
check(/CORS/.test(getEl('testLiveGrid').innerHTML),'CORS cause shown to user');

section('16. Raw response viewer and step-by-step fetch log');
mockSucceed(ALL_RAW,TEST_DATE,{rows:ALL_ROWS});
api.clearDiagnostics();
await api.testLiveConnection();
check(Object.keys(api.DIAG.raw).length===6,'raw bodies captured for all six symbols');
const rawSpy=api.DIAG.raw.SPY;
check(rawSpy.status===200&&rawSpy.bytes>0&&rawSpy.symbol==='SPY','raw response has status, size and retrieved symbol');
check(rawSpy.lines.split('\n').length<=200,'raw response capped at 200 lines');
check(api.SYMBOL_KEYS.every(k=>api.DIAG.raw[api.DATA_CONFIG.symbols[k]]&&api.DIAG.raw[api.DATA_CONFIG.symbols[k]].at instanceof Date),'raw responses carry retrieval timestamps');
const dbg=getEl('debugPanel').innerHTML;
check(/Fetch · SPY/.test(dbg)&&/HTTP 200/.test(dbg),'debug log shows symbol and HTTP status');
check(dbg.includes('worker.example.com')&&/URL https:\/\/worker\.example\.com/.test(dbg),'debug log shows actual worker endpoint');
check(/Time \d+ms/.test(dbg),'debug log shows latency');
check(/SUCCESS/.test(dbg),'debug log shows success badges');
check(getEl('debugInline').innerHTML===dbg,'inline debug panel mirrors the full log');

section('17. Transparency report, validation and health after a live score');
mockSucceed(ALL_RAW,OK_DATE,{rows:allLiveUpTo(OK_DATE)});
getEl('datePick').value=OK_DATE;
api.clearAuditLog();
await api.generateScore();
const diagResult=api.getLastResult();
check(diagResult&&diagResult.source==='live','generated score used all-live data');
const trans=getEl('transparencyBox').innerHTML;
check(/LIVE/.test(trans)&&/Yahoo Finance/.test(trans),'transparency report states live source and provider');
check(/ALL SYMBOLS VERIFIED/.test(trans),'verification status shown');
check(/Retrieval timestamp/.test(trans)&&/Market observation date/.test(trans),'timestamp and observation date shown');
check(/Confidence score/.test(trans)&&/100%/.test(trans),'confidence shown');
const val=getEl('validationTable').innerHTML;
check(/Requested symbol/.test(val)&&/Returned symbol/.test(val),'validation panel has requested/returned columns');
check((val.match(/PASSED/g)||[]).length===6,'all six validations passed');
check(/Reachable/.test(getEl('healthKpis').innerHTML),'provider health reports reachable');
check(api.DIAG.log[0].event==='Generate Score'&&api.DIAG.log[0].success,'successful score logged');
check(api.DIAG.log[0].score===diagResult.score.total,'audit log records the score');

section('18. Failure path: no misleading score, detailed message');
mockSucceed(new Set(['SPY','QQQ']),OK_DATE,{rows:allLiveUpTo(OK_DATE)});
api.clearAuditLog();
await api.generateScore();
check(/Mixed/.test(getEl('transparencyBox').innerHTML)||/MIXED/.test(getEl('transparencyBox').innerHTML),'partial source labelled MIXED in transparency report');
getEl('datePick').value='2099-01-01';
mockAllFail();
await api.generateScore();
const failHtml=getEl('genStatus').innerHTML;
check(/Fetch data failed/.test(failHtml),'failure header names the failing phase');
check(/live provider could not be reached/.test(failHtml),'detailed live-provider explanation visible');
check(/No usable market data/.test(failHtml),'explicit no-data warning');
check(api.DIAG.log[0].success===false&&api.DIAG.log[0].confidence===0,'failed attempt logged with zero confidence');

section('19. Audit log cap and local persistence');
for(let i=0;i<60;i++) api.appendAuditLog({event:'bulk',source:'live',success:true,reason:'',confidence:100,score:78});
check(api.DIAG.log.length===50,'audit log capped at 50 entries');
check(storageMap.has(api.DIAG_STORE.log),'audit log persisted in localStorage');
check(storageMap.has(api.DIAG_STORE.health),'provider health persisted in localStorage');
getEl('datePick').value=OK_DATE;

section('20. Missing chart library degrades gracefully');
const apiNoChart=factory(D, documentStub, windowStub, localStorageStub, getComputedStyleStub, null, fetchProxy, locationStub);
let chartThrew=false; try{ apiNoChart.getChart('chartHistory'); }catch(e){ chartThrew=true; }
check(!chartThrew,'missing ECharts does not throw');
check(/Chart library unavailable/.test(getEl('chartHistory').innerHTML),'chart failure explained in the UI');

section('21. file:// detection, plain-language messages, embedded-only mode');
const fileApi=factory(D, documentStub, windowStub, localStorageStub, getComputedStyleStub, echartsStub, fetchProxy, {protocol:'file:'});
check(fileApi.isFileMode()===true,'file:// protocol detected automatically');
check(/This dashboard is running directly from your computer[\s\S]{0,160}Browser security restrictions may prevent live market data retrieval/.test(HTML),'exact file:// notice present in built HTML');
check(/onclick="continueWithEmbedded\(\)"/.test(HTML)&&/>Continue with embedded data</.test(HTML),'embedded-data button offered');
check(/onclick="openDeploymentHelp\(\)"/.test(HTML)&&/>Open deployment instructions</.test(HTML),'deployment-instructions button offered');
check(getEl('fileModeBanner').style.display==='flex','banner shown in file mode');
check(getEl('hostedLinkBox').innerHTML.length>0,'Help tab renders hosted-link area');
check(/Launch Dashboard\.(bat|cmd)/.test(HTML)&&/GitHub Pages/.test(HTML),'ranked deployment instructions present in app');
const friendly=fileApi.friendlyUnavailableHtml();
check(/Live market quotes are currently unavailable/.test(friendly),'friendly unavailable headline');
check(/embedded historical dataset dated/.test(friendly)&&/2026/.test(friendly),'embedded dataset date stated');
check(/No action is required/.test(friendly),'explicit "no action required" message');
fileApi.continueWithEmbedded();
check(getEl('fileModeBanner').style.display==='none','banner dismissed after choice');
check(getEl('offlineBadge').style.display==='inline-flex','offline badge visible');
check(storageMap.get('etfEmbeddedOnly')==='1','embedded choice persisted locally');
let fetchCalls=0;
currentFetch=async()=>{ fetchCalls++; throw new Error('network should not be used'); };
getEl('datePick').value=OK_DATE;
await fileApi.generateScore();
check(fetchCalls===0,'embedded-only mode makes zero network calls');
const embeddedResult=fileApi.getLastResult();
check(embeddedResult&&embeddedResult.source==='embedded','embedded-only score attributed to embedded data');
fileApi.exitEmbeddedMode();
check(getEl('offlineBadge').style.display==='none','offline badge hidden after returning to live mode');

section('22. Provider status panel and per-symbol request debug fields');
currentFetch=async(url)=>{
  if(String(url).includes('/health')) return {ok:true,status:200,text:async()=>JSON.stringify({ok:true,version:'1.0.0',provider:'Yahoo Finance'})};
  throw new Error('unexpected URL '+url);
};
const ps=await api.probeProviderStatus();
check(ps.worker.state==='PASS'&&ps.yahoo.state==='FAIL'&&ps.embedded.state==='PASS','worker/yahoo/embedded status computed');
check(/PASS/.test(getEl('providerStatusBox').innerHTML)&&/Current provider in use/.test(getEl('providerStatusTable').innerHTML),'status cards and current provider rendered');
mockSucceed(new Set(['SPY','QQQ','^VIX','TLT','^TNX','^IRX']),TEST_DATE,{rows:ALL_ROWS});
await api.testLiveConnection();
const connGrid=getEl('testLiveGrid').innerHTML;
check(/Source:<\/span>\s*Worker/.test(connGrid)||/Source: Worker/.test(connGrid),'per-symbol source shown as Worker');
check(/URL Called:/.test(connGrid)&&connGrid.includes('worker.example.com'),'actual worker endpoint shown');
check(/Response:/.test(connGrid)&&/Latency:/.test(connGrid),'HTTP response and latency shown per symbol');

section('23. Score charts limited to trailing 1 year');
api.renderHistoryChart(D.n-1);
const histOpt=chartStub.lastOption;
check(histOpt&&histOpt.xAxis.data.length===api.ONE_YEAR_SESSIONS,'Date Explorer score chart shows 252 sessions');
check(histOpt.xAxis.data[0]===D.dates[D.n-api.ONE_YEAR_SESSIONS],'Date Explorer chart starts one year back');
api.renderTrendScoreChart(api.scoreTrendStats(D.n-1),D.n-1);
const trendOpt=chartStub.lastOption;
check(trendOpt&&trendOpt.xAxis.data.length===api.ONE_YEAR_SESSIONS,'Buy Score Trend chart shows 252 sessions');
check(trendOpt.xAxis.data[0]===D.dates[D.n-api.ONE_YEAR_SESSIONS],'trend chart starts one year back');

if(process.argv.includes('--live')){
  section('13. LIVE network smoke test (real Yahoo Finance)');
  const realFetch=(...a)=>fetch(...a);
  // Rebuild api with the real fetch swapped in via a mutable bridge.
  const saved=currentFetch;
  const savedWorkerBase=storageMap.get('etfWorkerBase');
  const savedDirect=api.DATA_CONFIG.directYahooEnabled;
  // This smoke test intentionally exercises direct Yahoo to prove the upstream
  // endpoint still works. Normal app operation uses the Worker, not this path.
  storageMap.set('etfWorkerBase','');
  api.DEPLOY_CONFIG.workerBase='';
  api.DATA_CONFIG.directYahooEnabled=true;
  currentFetch=realFetch;
  api.DATA_CONFIG.retries=1;
  const today=new Date().toISOString().slice(0,10);
  try{
    const r=await api.fetchMarketHistory('^VIX','2026-01-01',today);
    console.log('  ^VIX live rows='+r.rows.length+' verified='+r.symbolVerified+' last='+r.rows[r.rows.length-1].date);
    check(r.rows.length>0&&r.symbolVerified===true,'real Yahoo ^VIX fetch succeeded');
  api.clearDiagnostics();
  const all=await api.loadSeriesForDate(today);
  console.log('  full loadSeriesForDate('+today+') → source='+all.source+' live='+all.liveCount+' embedded='+all.embeddedCount+' missing='+all.missingCount);
  const realRaw=api.DIAG.raw['^VIX'];
  check(!!realRaw&&/"chart"/.test(realRaw.lines||''),'Raw Response Viewer captured a genuine Yahoo JSON payload');
  check(!!realRaw&&realRaw.symbol==='^VIX'&&realRaw.status===200,'raw response records real returned symbol and HTTP status');
  check(Object.keys(api.DIAG.raw).length===6,'raw responses stored for all six live symbols');
    api.SYMBOL_KEYS.forEach(k=>{
      const p=all.provenance[k];
      console.log('    '+k.padEnd(4)+' '+p.source.padEnd(9)+' rows='+String(all.series[k].length).padEnd(4)+
                  ' reported='+String(p.reportedSymbol)+' attempts='+p.attempts);
    });
    check(all.liveCount===6,'all six symbols actually fetched from Yahoo Finance');
    const li=api.calculateIndicators(all.series,today);
    api.validateIndicators(li.ind,all.provenance);
    const ls=api.computeScore(api.indicatorsToScoreInput(li.ind));
    const q=api.assessDataQuality(all,today,li.marketDate,5,7);
    console.log('  LIVE SCORE '+li.marketDate+': '+ls.total+' ['+api.bandOf(ls.total)+'] confidence='+q.confidence+'%');
    check(ls.total>=0&&ls.total<=100,'live score within 0–100');
    check(q.confidence===100,'live score has 100% live-data confidence');
  }catch(e){ console.log('  live fetch error: '+e.message); check(false,'real Yahoo fetch (network may be unavailable)'); }
  currentFetch=saved; api.DATA_CONFIG.retries=0;
  api.DATA_CONFIG.directYahooEnabled=savedDirect;
  if(savedWorkerBase===undefined) storageMap.delete('etfWorkerBase'); else storageMap.set('etfWorkerBase',savedWorkerBase);
}

console.log('\n=====================================================');
console.log('RESULT: '+pass+' passed, '+fail+' failed');
if(fail) console.log(results.join('\n'));
console.log('=====================================================');
process.exit(fail?1:0);
})().catch(e=>{ console.error('\nHARNESS ERROR:',e.stack); process.exit(2); });
