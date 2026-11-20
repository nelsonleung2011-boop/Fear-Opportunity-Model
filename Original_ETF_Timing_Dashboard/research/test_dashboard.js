/* Validation harness for the dashboard's data/indicator/score pipeline.
   Extracts the pure functions from the HTML and runs them against embedded data. */
const fs=require('fs');
const html=fs.readFileSync('ETF_Timing_Dashboard.html','utf8');
const data=fs.readFileSync('dashboard_data.json','utf8');
const s=html.indexOf('const D = ');
const e=html.indexOf('function compsFromArrays(i){', s);
let code=html.slice(s,e);
const nl=code.indexOf('\n');
code=code.slice(code.indexOf('\n',nl+1)+1); // drop 'const D=...' and 'const N=...' lines
// stub browser globals referenced but not used by tested functions
global.getComputedStyle=()=>({getPropertyValue:()=> '#000'});
global.localStorage={getItem:()=>null,setItem:()=>{}};
const api = new Function('D', 'const N = D.n;\n' + code + '\nreturn {N, calculateIndicators, computeScore, indicatorsToScoreInput, detectSignals, validateIndicators, getEmbeddedSeries, loadSeriesForDate, parseHistoryJSON, bandOf};')(JSON.parse(data));
const {N, calculateIndicators, computeScore, indicatorsToScoreInput, detectSignals, validateIndicators, getEmbeddedSeries, loadSeriesForDate, parseHistoryJSON, bandOf} = api;
const D = JSON.parse(data);

let pass=0, fail=0;
function check(cond,msg){ if(cond){pass++;} else {fail++; console.log('  FAIL:',msg);} }

console.log('=== A. No-look-ahead: recompute score from raw embedded series vs stored score_history ===');
let maxDiff=0, mismatches=0;
const startI=D.dates.indexOf('2003-04-01');
for(let i=startI;i<N;i+=37){
  const dt=D.dates[i];
  const series={};
  ['SPY','QQQ','VIX','TLT','TNX','IRX'].forEach(k=>{
    series[k]=[];
    const arr=D[k==='SPY'?'SPY':k==='QQQ'?'QQQ':k==='VIX'?'VIX':k==='TLT'?'TLT':k==='TNX'?'TNX':'IRX'];
    for(let j=0;j<=i;j++) if(typeof arr[j]==='number'&&isFinite(arr[j])) series[k].push({date:D.dates[j],close:arr[j]});
  });
  const {ind,marketDate}=calculateIndicators(series,dt);
  const sc=computeScore(indicatorsToScoreInput(ind));
  const diff=Math.abs(sc.total-D.opp_score[i]);
  maxDiff=Math.max(maxDiff,diff);
  if(diff>0.01){ mismatches++; if(mismatches<6) console.log('  mismatch',dt,'recomputed',sc.total,'stored',D.opp_score[i],JSON.stringify(sc)); }
  check(marketDate===dt,'market date '+dt+' got '+marketDate);
}
console.log('  checked ~'+Math.ceil((N-startI)/37)+' dates, max score diff='+maxDiff+', mismatches='+mismatches);
check(mismatches===0,'score_history reproducibility');

console.log('\n=== B. Trading-date resolution (weekend / holiday / normal) ===');
function embeddedSeriesUpTo(date){
  const s2={};
  ['SPY','QQQ','VIX','TLT','TNX','IRX'].forEach(k=>{ s2[k]=getEmbeddedSeries(k==='SPY'?'SPY':k==='QQQ'?'QQQ':k==='VIX'?'VIX':k==='TLT'?'TLT':k==='TNX'?'TNX':'IRX',date); });
  return s2;
}
const dateCases=[
  ['2024-06-14','2024-06-14','normal weekday'],
  ['2026-11-15','2026-11-13','Sunday'],
  ['2024-12-25','2024-12-24','Christmas holiday'],
  ['2020-03-23','2020-03-23','COVID crash weekday'],
  ['2008-09-17','2008-09-17','GFC weekday'],
  ['2022-10-12','2022-10-12','2022 bear weekday'],
];
dateCases.forEach(([req,exp,label])=>{
  const {ind,marketDate}=calculateIndicators(embeddedSeriesUpTo(req),req);
  console.log('  '+label+': requested '+req+' -> market '+marketDate+' score='+computeScore(indicatorsToScoreInput(ind)).total);
  check(marketDate===exp,label+' expected '+exp+' got '+marketDate);
});

console.log('\n=== C. Score integrity across key dates ===');
['2008-09-17','2020-03-23','2022-10-12','2026-11-13'].forEach(dt=>{
  const r=calculateIndicators(embeddedSeriesUpTo(dt),dt);
  validateIndicators(r.ind);
  const sc=computeScore(indicatorsToScoreInput(r.ind));
  const sum=sc.trend+sc.dip+sc.vix+sc.macro+sc.season;
  check(sc.total===sum,'sum '+dt);
  check(sc.total>=0&&sc.total<=100,'range '+dt);
  check(sc.trend<=25&&sc.dip<=25&&sc.vix<=25&&sc.macro<=15&&sc.season<=10,'component caps '+dt);
  const sig=detectSignals(r.ind,r.prev);
  console.log('  '+dt+' score='+sc.total+' ['+bandOf(sc.total)+'] comps='+JSON.stringify([sc.trend,sc.dip,sc.vix,sc.macro,sc.season])+' active='+Object.keys(sig).filter(k=>sig[k].active).join(','));
});

console.log('\n=== D. Active vs fresh distinction ===');
// 2020-03-16: VIX crossed above 35 (prior day 2020-03-13 VIX ~61? check actual crossing)
['2020-03-16','2020-03-23','2026-11-13'].forEach(dt=>{
  const r=calculateIndicators(embeddedSeriesUpTo(dt),dt);
  const sig=detectSignals(r.ind,r.prev);
  console.log('  '+dt+' VIX active='+sig.vix35.active+' fresh35cross='+sig.vix35cross.active+' prevVIX='+(r.prev?r.prev.vix:'n/a')+' VIX='+r.ind.vix);
  check(typeof sig.vix35.active==='boolean'&&typeof sig.vix35cross.active==='boolean','signal booleans '+dt);
});

console.log('\n=== E. Online failure -> embedded fallback (simulate fetch rejection) ===');
global.fetch=async()=>{ throw new Error('simulated CORS block'); };
(async()=>{
  const out=await loadSeriesForDate('2020-03-23');
  console.log('  fallback source='+out.source+' error='+out.error);
  check(out.source==='embedded','embedded fallback used');
  const out2=await loadSeriesForDate('2030-01-01');
  check(out2.source==='error','no fabricated data for out-of-range date');
  console.log('  out-of-range source='+out2.source+' error='+out2.error);

  console.log('\n=== F. Yahoo JSON parse validation ===');
  const fake={chart:{result:[{timestamp:[1700000000,1700086400],indicators:{quote:[{close:[100,101]}]}}]}};
  const parsed=parseHistoryJSON(fake,'test');
  check(parsed.length===2&&parsed[0].date==='2023-11-14','parse yahoo shape');
  let threw=false; try{ parseHistoryJSON({bad:1},'test'); }catch(e){ threw=true; } check(threw,'reject bad JSON shape');

  console.log('\n=== RESULT: '+pass+' passed, '+fail+' failed ===');
  process.exit(fail?1:0);
})();
