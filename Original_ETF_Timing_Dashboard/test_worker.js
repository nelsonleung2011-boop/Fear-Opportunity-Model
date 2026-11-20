/* =====================================================================
   Cloudflare Worker validation harness (no network, no Cloudflare account)
   Run: node test_worker.js
   ===================================================================== */
const fs = require('fs');

let pass=0, fail=0;
function check(cond,msg){ if(cond){ pass++; console.log('  OK  '+msg); } else { fail++; console.log('  FAIL '+msg); } }
function section(t){ console.log('\n=== '+t+' ==='); }

// Convert the ES-module Worker into a callable factory for a plain Node test.
const source=fs.readFileSync('worker.js','utf8').replace('export default {','const __WORKER = {');
const cacheStore=new Map();
const cachesMock={default:{
  async match(req){ const r=cacheStore.get(req.url); return r?r.clone():undefined; },
  async put(req,resp){ cacheStore.set(req.url,resp.clone()); },
  async delete(req){ return cacheStore.delete(req.url); }
}};

let upstreamCalls=0;
let upstreamMode='ok';
const fetchMock=async(url)=>{
  upstreamCalls++;
  if(upstreamMode==='fail500'&&upstreamCalls===1){
    return new Response('upstream error',{status:503});
  }
  const u=String(url);
  const sym=decodeURIComponent(/%5E|%24/.test(u)?u.split('/').pop().split('?')[0].replace('%5E','^'):u.split('/').pop().split('?')[0]);
  const last='2026-11-16';
  const payload={chart:{error:null,result:[{
    meta:{symbol:sym,currency:'USD'},
    timestamp:[Math.floor(Date.parse(last+'T00:00:00Z')/1000)-86400,Math.floor(Date.parse(last+'T00:00:00Z')/1000)],
    indicators:{quote:[{close:[100,101]}]}
  }]}};
  return new Response(JSON.stringify(payload),{status:200,headers:{'Content-Type':'application/json'}});
};

const factory=new Function(
  'caches','fetch','Response','Headers','Request','URL','setTimeout','clearTimeout','console',
  source+'\nreturn __WORKER;'
);
const worker=factory(cachesMock,fetchMock,Response,Headers,Request,URL,setTimeout,clearTimeout,console);
const base='https://worker.example.com';

(async()=>{
section('1. Health endpoint and CORS');
cacheStore.clear();
let resp=await worker.fetch(new Request(base+'/health'),{}, {});
check(resp.status===200,'health returns HTTP 200');
check(resp.headers.get('Access-Control-Allow-Origin')==='*','CORS allow-origin header present');
let body=await resp.json();
check(body.ok===true&&body.provider==='Yahoo Finance','health JSON identifies provider');
check(Array.isArray(body.symbols)&&body.symbols.length===6,'all six symbols declared');
resp=await worker.fetch(new Request(base+'/chart?symbol=SPY&period1=1&period2=2',{method:'OPTIONS'}),{},{});
check(resp.status===204,'OPTIONS preflight returns 204');

section('2. Chart endpoint, symbol verification and clean JSON');
cacheStore.clear(); upstreamCalls=0; upstreamMode='ok';
const chartUrl=base+'/chart?symbol=%5EVIX&period1=1704067200&period2=1735689600';
resp=await worker.fetch(new Request(chartUrl),{},{});
check(resp.status===200,'chart returns HTTP 200');
check(resp.headers.get('Content-Type').includes('application/json'),'clean JSON content type');
check(resp.headers.get('Cache-Control').includes('max-age=600'),'10-minute cache header');
body=await resp.json();
check(body.ok===true&&body.symbol==='^VIX','requested symbol echoed and verified');
check(body.data.length===2&&body.data[0].date==='2026-11-15','normalised rows sorted by date');
check(body.provider==='Yahoo Finance'&&body.via==='Cloudflare Worker','provider attribution present');
check(body.cached===false&&body.attempts===1,'first response is a cache MISS');

section('3. Cache HIT and no extra upstream call');
const callsAfterFirst=upstreamCalls;
resp=await worker.fetch(new Request(chartUrl),{},{});
body=await resp.json();
check(resp.headers.get('X-Worker-Cache')==='HIT','second response is cache HIT');
check(body.cached===true,'payload marks cached=true');
check(upstreamCalls===callsAfterFirst,'no extra Yahoo request on cache hit');

section('4. Retry on transient upstream 503');
cacheStore.clear(); upstreamCalls=0; upstreamMode='fail500';
resp=await worker.fetch(new Request(base+'/chart?symbol=SPY&period1=1704067200&period2=1735689600'),{},{});
check(resp.status===200,'succeeds after a transient 503');
body=await resp.json();
check(body.attempts===2,'two upstream attempts recorded');

section('5. Meaningful error codes');
cacheStore.clear(); upstreamCalls=0; upstreamMode='ok';
resp=await worker.fetch(new Request(base+'/chart?symbol=AAPL&period1=1&period2=2'),{},{});
check(resp.status===400,'unsupported symbol returns HTTP 400');
body=await resp.json();
check(body.ok===false&&body.error.code==='SYMBOL_NOT_ALLOWED','SYMBOL_NOT_ALLOWED code returned');
resp=await worker.fetch(new Request(base+'/chart?symbol=SPY&period1=bad&period2=2'),{},{});
body=await resp.json();
check(body.error.code==='INVALID_DATE_RANGE','INVALID_DATE_RANGE code returned');
resp=await worker.fetch(new Request(base+'/unknown'),{},{});
body=await resp.json();
check(body.error.code==='ROUTE_NOT_FOUND','ROUTE_NOT_FOUND code returned');

console.log('\n=====================================================');
console.log('WORKER RESULT: '+pass+' passed, '+fail+' failed');
console.log('=====================================================');
process.exit(fail?1:0);
})().catch(e=>{ console.error('WORKER HARNESS ERROR:',e.stack); process.exit(2); });
