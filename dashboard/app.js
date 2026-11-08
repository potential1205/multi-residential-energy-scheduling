'use strict';
const $=id=>document.getElementById(id), colors=['#447552','#b98b52','#47698a','#8d6aa0','#88a45b','#b4533f'];
const state={runs:[],selected:new Set(),active:null,details:[],paused:false,busy:false};
const finite=v=>v!==null&&v!==undefined&&Number.isFinite(Number(v));
const fmt=(n,d=2)=>finite(n)?Number(n).toLocaleString('en-US',{maximumFractionDigits:d,minimumFractionDigits:d}):'—';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const avg=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:null;
const sec=n=>finite(n)?n<60?`${Math.round(n)}s`:n<3600?`${Math.floor(n/60)}m ${Math.round(n%60)}s`:`${Math.floor(n/3600)}h ${Math.floor(n%3600/60)}m`:'—';
const meanField=(e,k)=>avg((e.clients||[]).map(c=>c[k]).filter(finite).map(Number));
const STATUS={running:['실행 중','프로세스가 살아 있고 진행 상황을 보고 중'],stalled:['응답 없음','프로세스는 있으나 10분간 보고 없음'],
 queued:['대기','suite에 계획됨, 아직 시작 전'],completed:['완료','정상 종료'],stopped:['중단','요청으로 멈춤, 부분 결과 저장됨'],
 failed:['실패','오류로 종료'],interrupted:['비정상 종료','실행 중으로 기록됐지만 프로세스가 없음'],unknown:['알 수 없음','상태 파일 없음']};
const statusOf=r=>STATUS[r.status?.status]?r.status.status:'unknown';
const badge=r=>{const s=statusOf(r),[label,help]=STATUS[s],detail=r.status?.error||r.status?.reason||'';return `<span class="badge s-${s}" title="${esc(help+(detail?' · '+detail:''))}"><i class="dot"></i>${esc(label)}</span>`;};
const xs=e=>$('axis').value==='elapsed'?e.elapsed:e.day;
const epochsOf=c=>c.training_episodes||c.episodes||(c.iterations||0)*30;
function smooth(values,n){let sum=0;return values.map((p,i)=>{sum+=p.y;if(i>=n)sum-=values[i-n].y;return {...p,y:sum/Math.min(i+1,n)};});}
function curves(run,type,key,scale=1,window=1){const values=run.events.filter(e=>e.type===type).map(e=>{const v=type==='validation'?e[key]:meanField(e,key);return {x:xs(e),y:finite(v)?v*scale:null};}).filter(p=>finite(p.x)&&finite(p.y));return smooth(values,window);}
// Cost events may arrive binned by the server (`span` days per point); convert a window in days to points.
const points=(run,days)=>Math.max(1,Math.round(days/(run.events.find(e=>e.type==='cost')?.span||1)));

function ticks(min,max,count=5){
  const step0=(max-min)/Math.max(count-1,1),mag=10**Math.floor(Math.log10(step0||1)),step=[1,2,2.5,5,10].map(m=>m*mag).find(s=>s>=step0)||mag*10;
  const out=[];for(let v=Math.ceil(min/step)*step;v<=max+step*1e-9;v+=step)out.push(+v.toPrecision(12));
  // Narrow charts: fall back to a finer step rather than showing a single tick.
  return out.length<2&&count>2?ticks(min,max,count+2):out;
}
// Tick labels use just enough decimals to tell neighbouring ticks apart.
const tickLabel=(n,step)=>{const a=Math.abs(n),big=a>=1e6?[1e6,'M']:a>=1e4?[1e3,'k']:[1,''],d=Math.max(0,Math.min(4,-Math.floor(Math.log10(step/big[0])+1e-9)));return `${(n/big[0]).toFixed(d)}${big[1]}`;};
const compact=n=>{const a=Math.abs(n);return a>=1e6?`${+(n/1e6).toFixed(1)}M`:a>=1e3?`${+(n/1e3).toFixed(1)}k`:a>=10?`${Math.round(n)}`:a>=1?`${+n.toFixed(2)}`:a===0?'0':`${+n.toPrecision(2)}`;};
function plot(id,series,{log=false,bands=[]}={}){
  const el=$(id),clean=series.map(s=>({...s,data:s.data.filter(p=>finite(p.x)&&finite(p.y)&&(!log||p.y>0))})).filter(s=>s.data.length);
  if(!clean.length){el.innerHTML='<div class="empty">아직 기록된 값이 없습니다</div>';el.onmousemove=null;return;}
  const width=Math.max(el.clientWidth,260),height=Math.max(el.clientHeight-20,120),pad={l:44,r:8,t:8,b:22};
  const T=y=>log?Math.log10(y):y,pts=clean.flatMap(s=>s.data);
  let xmin=Math.min(...pts.map(p=>p.x)),xmax=Math.max(...pts.map(p=>p.x));if(xmax===xmin)xmax=xmin+1;
  const ys=pts.map(p=>T(p.y));bands.forEach(b=>b.data.forEach(p=>ys.push(p.low,p.high)));
  let ymin=Math.min(...ys),ymax=Math.max(...ys);if(ymin===ymax){ymin-=Math.abs(ymin)*.1||.1;ymax+=Math.abs(ymax)*.1||.1;}
  const room=(ymax-ymin)*.06;ymin-=room;ymax+=room;
  const X=x=>pad.l+(x-xmin)/(xmax-xmin)*(width-pad.l-pad.r),Y=y=>height-pad.b-(y-ymin)/(ymax-ymin)*(height-pad.t-pad.b);
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(id)}">`;
  let yt,ylab;
  if(log){const decades=[];for(let k=Math.ceil(ymin);k<=Math.floor(ymax);k++)decades.push(k);
    const fine=[];for(let k=Math.floor(ymin);k<=Math.ceil(ymax);k++)[1,2,5].forEach(m=>{const v=Math.log10(m)+k;if(v>=ymin&&v<=ymax)fine.push(v);});
    yt=decades.length>=3?decades:fine;ylab=v=>compact(10**v);}
  else{yt=ticks(ymin,ymax,5);const step=yt.length>1?yt[1]-yt[0]:Math.abs(ymax-ymin)/2||1;ylab=v=>tickLabel(v,step);}
  for(const y of yt)svg+=`<line x1="${pad.l}" x2="${width-pad.r}" y1="${Y(y)}" y2="${Y(y)}" stroke="#eef0ea"/><text x="${pad.l-6}" y="${Y(y)+3.5}" text-anchor="end" fill="#959d8d" font-size="10">${ylab(y)}</text>`;
  const xt=ticks(xmin,xmax,Math.max(3,Math.min(7,Math.floor(width/110)))),xstep=xt.length>1?xt[1]-xt[0]:(xmax-xmin)/2;
  for(const x of xt)svg+=`<text x="${X(x)}" y="${height-5}" text-anchor="middle" fill="#959d8d" font-size="10">${tickLabel(x,xstep)}</text>`;
  svg+=`<line x1="${pad.l}" x2="${width-pad.r}" y1="${height-pad.b}" y2="${height-pad.b}" stroke="#dfe3d8"/>`;
  bands.forEach(b=>{const p=[...b.data.map(q=>`${X(q.x)},${Y(q.high)}`),...b.data.slice().reverse().map(q=>`${X(q.x)},${Y(q.low)}`)].join(' ');svg+=`<polygon points="${p}" fill="#9aa596" opacity=".16"/>`;});
  clean.forEach(s=>{const stride=Math.max(1,Math.ceil(s.data.length/1500));const d=s.data.filter((_,i)=>i%stride===0||i===s.data.length-1);
    svg+=`<path d="${d.map((p,i)=>`${i?'L':'M'}${X(p.x).toFixed(1)},${Y(T(p.y)).toFixed(1)}`).join('')}" stroke="${s.color}" fill="none" stroke-width="${s.width||1.6}" stroke-dasharray="${s.dash||''}" stroke-linejoin="round"/>`;
    if(s.marks)d.forEach(p=>svg+=`<circle cx="${X(p.x).toFixed(1)}" cy="${Y(T(p.y)).toFixed(1)}" r="2.2" fill="${s.color}"/>`);});
  svg+=`<line class="cursor" x1="0" x2="0" y1="${pad.t}" y2="${height-pad.b}" stroke="#b9c1b2" stroke-dasharray="3 3" visibility="hidden"/></svg>`;
  el.innerHTML=svg+'<div class="legend">'+clean.map(s=>`<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('')+'</div>';
  const cursor=el.querySelector('.cursor');
  el.onmouseleave=()=>{$('tooltip').style.display='none';cursor.setAttribute('visibility','hidden');};
  el.onmousemove=e=>{const r=el.getBoundingClientRect(),px=(e.clientX-r.left)*width/r.width;if(px<pad.l||px>width-pad.r){el.onmouseleave();return;}
    const x=xmin+(px-pad.l)/(width-pad.l-pad.r)*(xmax-xmin);cursor.setAttribute('x1',px);cursor.setAttribute('x2',px);cursor.setAttribute('visibility','visible');
    const lines=clean.map(s=>{let lo=0,hi=s.data.length-1;while(lo<hi){const m=(lo+hi)>>1;if(s.data[m].x<x)lo=m+1;else hi=m;}return `${s.name}  ${fmt(s.data[lo].y,3)}`;});
    const tip=$('tooltip');tip.textContent=`x = ${fmt(x,0)}\n`+lines.join('\n');tip.style.display='block';tip.style.left=Math.min(e.clientX+14,innerWidth-280)+'px';tip.style.top=Math.max(8,e.clientY-60)+'px';};
}

function seedBands(runs,days){
  const keys=['algorithm','source','scenario','epsilon','gamma','lr','loss','training_episodes','forecast','clients','co','target_hours','width','batch_size','updates_per_day','cost_scale','new_clients'];
  const groups=new Map();runs.forEach(r=>{const k=JSON.stringify(keys.map(x=>r.config[x]));groups.set(k,[...(groups.get(k)||[]),r]);});
  return [...groups.values()].filter(g=>new Set(g.map(r=>r.config.seed)).size>1).map(g=>{
    const maps=g.map(r=>new Map(curves(r,'cost','cost',.001,points(r,days)).map(p=>[p.x,p.y])));const data=[];
    for(const x of maps[0].keys()){const v=maps.map(m=>m.get(x));if(v.every(finite)){const m=avg(v),sd=Math.sqrt(avg(v.map(y=>(y-m)**2)));data.push({x,low:m-sd,high:m+sd});}}
    return {data};});
}
const shortName=id=>id.split('/').at(-1);
const hyper=c=>{const r=c.resolved||{},parts=[];
  if(c.scenario&&c.algorithm!=='Co-TDAS')parts.push(`시나리오 ${c.scenario}`);if(c.algorithm==='Co-TDAS')parts.push(`EMS ${c.clients}`);
  parts.push(`seed ${c.seed}`);
  const lr=r.learning_rate??c.lr;if(finite(lr))parts.push(`lr ${lr}`);
  parts.push(`업데이트 ${c.updates_per_day||1}/일`);if(c.cost_scale&&c.cost_scale!==1)parts.push(`비용 ÷${c.cost_scale}`);
  parts.push(`target ${c.target_hours||20}h`);parts.push(`폭 ${c.width||512}`);if(c.batch_size&&c.batch_size!==32)parts.push(`batch ${c.batch_size}`);
  if(c.loss&&c.loss!=='huber')parts.push(c.loss.toUpperCase());return parts.join(' · ');};

function render(){
  const active=state.details.find(r=>r.id===state.active)||state.details[0];if(!active)return;
  const days=Number($('smooth').value),w=points(active,days),events=active.events,train=events.filter(e=>e.type==='train'),last=train.at(-1);
  const costs=events.filter(e=>e.type==='cost'),val=events.filter(e=>e.type==='validation').at(-1),all=state.details,c=active.config;
  $('active-name').textContent=shortName(active.id);$('active-status').innerHTML=badge(active);
  $('active-meta').textContent=`${c.algorithm||''} · ${hyper(c)}`;
  const done=last?.day||active.status?.day||costs.at(-1)?.day||0,total=epochsOf(c)||1;
  $('progress').textContent=`${fmt(Math.min(100,done/total*100),1)}%`;$('progress-bar').style.width=`${Math.min(100,done/total*100)}%`;$('progress-caption').textContent=`${fmt(done,0)} / ${fmt(total,0)} 일`;
  $('validation').textContent=val?`$${fmt(val.daily*.001,3)}`:'—';$('validation-caption').textContent=val?`최저 $${fmt(active.best_validation?.daily*.001,3)} · day ${fmt(active.best_validation?.day,0)}`:'검증 전';
  const cost=avg(costs.slice(-w).map(e=>meanField(e,'cost')));$('train-cost').textContent=finite(cost)?`$${fmt(cost*.001,3)}`:'—';$('train-caption').textContent=`최근 ${days}일 평균 · ε ${c.resolved?.epsilon??c.epsilon}`;
  const violations=last?.clients?.reduce((s,x)=>s+(x.violations||0),0);$('violations').textContent=finite(violations)?fmt(violations,0):'—';$('violations').style.color=violations?'var(--red)':'';
  const compare=(type,key,scale=1,win=r=>1)=>all.map((r,i)=>({name:shortName(r.id),color:colors[i%colors.length],marks:type==='validation',data:curves(r,type,key,scale,win(r))}));
  plot('cost-chart',compare('cost','cost',.001,r=>points(r,days)),{bands:seedBands(all,days)});
  plot('val-chart',compare('validation','daily',.001));
  plot('composition-chart',[['tou','TOU'],['dc','DC']].map(([k,n],i)=>({name:n,color:colors[i],data:curves(active,'cost',k,.001,w)})));
  plot('loss-chart',[['loss','loss'],['td_abs','|TD 오차|']].map(([k,n],i)=>({name:n,color:colors[i],data:curves(active,'train',k)})),{log:$('log-loss').checked});
  plot('q-chart',[['q_mean','Q(s,a) 평균'],['target_mean','target 평균']].map(([k,n],i)=>({name:n,color:colors[i],dash:i?'4 3':'',data:curves(active,'train',k)})));
  const ids=last?.clients?.map(x=>x.client)||[0],client=ids.includes(Number($('client').value))?Number($('client').value):ids[0];
  if($('client').options.length!==ids.length)$('client').innerHTML=ids.map(i=>`<option value="${i}">EMS ${i+1}</option>`).join('');
  $('client').hidden=ids.length<2;
  const schedule=last?.schedule?.[String(client)]||[],row=last?.clients?.find(x=>x.client===client);
  plot('schedule-chart',[['battery','배터리'],['action','구매'],['load','수요'],['generation','발전']].map(([k,n],i)=>({name:n,color:colors[i],data:schedule.map(p=>({x:p.hour,y:p[k]*.1}))})));
  $('schedule-caption').textContent=`day ${fmt(last?.day,0)} · kWh · 시간대별`;
  $('peak').textContent=row?`${fmt(row.peak*.1,1)} kWh`:'—';$('battery').textContent=row?`${fmt(row.final_battery*.1,1)} kWh`:'—';
  $('eta').textContent=`${sec(last?.elapsed||active.status?.seconds)} / ${statusOf(active)==='running'?sec(last?.eta_seconds):'—'}`;
  $('throughput').textContent=finite(last?.days_per_second)?`${fmt(last.days_per_second,0)} 일/초`:'—';
  $('config').textContent=JSON.stringify(c,null,2);
}

function statusCounts(runs){const n={};runs.forEach(r=>{const s=statusOf(r);n[s]=(n[s]||0)+1;});return Object.keys(STATUS).filter(s=>n[s]).map(s=>`${STATUS[s][0]} ${n[s]}`).join(' · ');}
function registry(){
  const q=$('search').value.trim().toLowerCase();
  const runs=state.runs.filter(r=>(!q||r.id.toLowerCase().includes(q))&&(!$('algorithm').value||r.config.algorithm===$('algorithm').value)&&(!$('status').value||statusOf(r)===$('status').value));
  $('run-count').innerHTML=`${runs.length}개 · ${statusCounts(runs)}`+(state.selected.size>1?` <button id="clear-compare" class="link">비교 ${state.selected.size}개 해제</button>`:'');
  if($('clear-compare'))$('clear-compare').onclick=()=>{state.selected=new Set([state.active]);refresh(true);};
  $('runs').innerHTML=runs.map(r=>{const total=epochsOf(r.config),s=statusOf(r),queued=!!r.queued,done=r.latest?.day||r.status?.day||(s==='completed'?total:0),pct=total?Math.min(100,100*done/total):0,name=queued?r.id.slice(7):shortName(r.id);
    return `<li class="${r.id===state.active?'active':''} ${queued?'queued':''}"><input type="checkbox" aria-label="${esc(name)} 비교" data-id="${esc(r.id)}" ${state.selected.has(r.id)?'checked':''} ${queued?'disabled':''}>
      <button class="run-name" data-active="${esc(r.id)}" title="${esc(r.id)}" ${queued?'disabled':''}>${esc(name)}</button>
      <span class="run-value" title="최저 검증 비용 $/일">${finite(r.best_validation?.daily)?'$'+fmt(r.best_validation.daily*.001,3):''}</span>
      <div class="run-sub">${badge(r)}<div class="mini-track"><div class="mini-bar s-${s}" style="width:${pct.toFixed(1)}%"></div></div><span>${compact(done)}/${compact(total)}</span></div></li>`;}).join('')||'<li class="muted small">조건에 맞는 실행이 없습니다</li>';
  $('runs').querySelectorAll('input').forEach(el=>el.onchange=()=>{if(el.checked){if(state.selected.size>=6){el.checked=false;return;}state.selected.add(el.dataset.id);}else{state.selected.delete(el.dataset.id);if(state.active===el.dataset.id)state.active=[...state.selected][0]||null;}refresh(true);});
  $('runs').querySelectorAll('.run-name').forEach(el=>el.onclick=()=>{state.active=el.dataset.active;state.selected.add(state.active);refresh(true);});
}
async function refresh(force=false){
  if((state.paused&&!force)||state.busy)return;state.busy=true;
  try{
    const response=await fetch('/api/runs');if(!response.ok)throw Error(response.status);state.runs=await response.json();
    const started=state.runs.filter(r=>!r.queued),valid=new Set(started.map(r=>r.id));
    state.selected=new Set([...state.selected].filter(id=>valid.has(id)));
    if(!state.selected.size&&started.length)state.selected.add(started[0].id);
    if(!state.selected.has(state.active))state.active=[...state.selected][0]||null;
    state.details=await Promise.all([...state.selected].slice(0,6).map(async id=>{const r=await fetch('/api/run?id='+encodeURIComponent(id));if(!r.ok)throw Error(r.status);return r.json();}));
    registry();render();
    $('connection').textContent='실시간';$('connection').className='badge';$('updated').textContent=new Date().toLocaleTimeString('ko-KR')+' 갱신';
  }catch(error){$('connection').textContent='연결 끊김';$('connection').className='badge error';$('updated').textContent=String(error);}
  finally{state.busy=false;}
}
$('pause').onclick=()=>{state.paused=!state.paused;$('pause').textContent=state.paused?'다시 시작':'일시정지';$('connection').textContent=state.paused?'일시정지됨':'실시간';if(!state.paused)refresh();};
['algorithm','status','search'].forEach(id=>$(id).oninput=registry);['smooth','axis','client','log-loss'].forEach(id=>$(id).onchange=render);
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(render,120);});
refresh();setInterval(refresh,3000);
