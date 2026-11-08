'use strict';
const $=id=>document.getElementById(id), colors=['#447552','#b98b52','#6486a0','#9673a4','#88a45b','#b96357'];
const state={runs:[],selected:new Set(),active:null,details:[],paused:false,busy:false};
const finite=v=>v!==null&&v!==undefined&&Number.isFinite(Number(v));
const fmt=(n,d=2)=>finite(n)?Number(n).toLocaleString('en-US',{maximumFractionDigits:d,minimumFractionDigits:d}):'—';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const avg=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:null;
const sec=n=>finite(n)?n<60?`${Math.round(n)}s`:n<3600?`${Math.floor(n/60)}m ${Math.round(n%60)}s`:`${Math.floor(n/3600)}h ${Math.floor(n%3600/60)}m`:'—';
const meanField=(e,k)=>avg((e.clients||[]).map(c=>c[k]).filter(finite).map(Number));
const xs=e=>$('axis').value==='elapsed'?e.elapsed:e.day;
function smooth(values,n){let sum=0;return values.map((p,i)=>{sum+=p.y;if(i>=n)sum-=values[i-n].y;return {...p,y:sum/Math.min(i+1,n)};});}
function curves(run,type,key,scale=1,window=1){const values=run.events.filter(e=>e.type===type).map(e=>{const v=type==='validation'?e[key]:meanField(e,key);return {x:xs(e),y:finite(v)?v*scale:null};}).filter(p=>finite(p.x)&&finite(p.y));return smooth(values,window);}
function plot(id,series,{log=false,bands=[]}={}){
  const el=$(id),clean=series.map(s=>({...s,data:s.data.filter(p=>finite(p.x)&&finite(p.y)&&(log?p.y>0:true))})).filter(s=>s.data.length);
  if(!clean.length){el.innerHTML='<div class="empty">아직 기록된 지표가 없습니다.</div>';return;}
  const width=Math.max(el.clientWidth,280),height=el.clientHeight-21,pad={l:48,r:14,t:10,b:28};
  const pts=clean.flatMap(s=>s.data),xp=pts.map(p=>p.x);let xmin=Math.min(...xp),xmax=Math.max(...xp);if(xmax===xmin)xmax=xmin+1;
  const trans=y=>log?Math.log10(y):y;const yp=pts.map(p=>trans(p.y));bands.forEach(b=>b.data.forEach(p=>{yp.push(p.low,p.high);}));let ymin=Math.min(...yp),ymax=Math.max(...yp);if(ymin===ymax){ymin-=Math.max(Math.abs(ymin)*.1,.1);ymax+=Math.max(Math.abs(ymax)*.1,.1);}const extra=(ymax-ymin)*.08;ymin-=extra;ymax+=extra;
  const X=x=>pad.l+(x-xmin)/(xmax-xmin)*(width-pad.l-pad.r),Y=y=>height-pad.b-(y-ymin)/(ymax-ymin)*(height-pad.t-pad.b);
  const compact=n=>Math.abs(n)>=1e6?`${fmt(n/1e6,1)}m`:Math.abs(n)>=1000?`${fmt(n/1000,1)}k`:Math.abs(n)<.1&&n!==0?Number(n).toPrecision(2):fmt(n,Math.abs(n)<10?2:0);
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${id}">`;
  for(let i=0;i<5;i++){const y=ymin+(ymax-ymin)*i/4;svg+=`<line x1="${pad.l}" x2="${width-pad.r}" y1="${Y(y)}" y2="${Y(y)}" stroke="#edf0e8"/><text x="${pad.l-9}" y="${Y(y)+3}" text-anchor="end" fill="#8f9b87" font-size="9">${compact(log?10**y:y)}</text>`;}
  for(let i=0;i<5;i++){const x=xmin+(xmax-xmin)*i/4;svg+=`<text x="${X(x)}" y="${height-8}" text-anchor="middle" fill="#8f9b87" font-size="9">${compact(x)}</text>`;}
  bands.forEach(b=>{const path=[...b.data.map(p=>`${X(p.x)},${Y(p.high)}`),...b.data.slice().reverse().map(p=>`${X(p.x)},${Y(p.low)}`)].join(' ');svg+=`<polygon points="${path}" fill="#9da99b" opacity=".17"/>`;});
  clean.forEach(s=>{const stride=Math.max(1,Math.ceil(s.data.length/1200));let sampled=s.data.filter((_,i)=>i%stride===0);if(sampled.at(-1)!==s.data.at(-1))sampled.push(s.data.at(-1));const path=sampled.map((p,i)=>`${i?'L':'M'}${X(p.x).toFixed(2)},${Y(trans(p.y)).toFixed(2)}`).join(' ');svg+=`<path d="${path}" stroke="${s.color}" fill="none" stroke-width="1.7" stroke-linejoin="round"/>`;});
  svg+='</svg><div class="chart-legend">'+clean.map(s=>`<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('')+'</div>';el.innerHTML=svg;
  el.onmousemove=e=>{const rect=el.getBoundingClientRect(),x=xmin+(e.clientX-rect.left-pad.l)/(width-pad.l-pad.r)*(xmax-xmin);const lines=clean.map(s=>{let lo=0,hi=s.data.length-1;while(lo<hi){let mid=(lo+hi)>>1;if(s.data[mid].x<x)lo=mid+1;else hi=mid;}const p=s.data[lo];return `${s.name}: ${fmt(p.y,3)} · x=${fmt(p.x,0)}`;});$('tooltip').textContent=lines.join('\n');$('tooltip').style.display='block';$('tooltip').style.left=Math.min(e.clientX+14,innerWidth-300)+'px';$('tooltip').style.top=Math.max(8,e.clientY-75)+'px';};
  el.onmouseleave=()=>{$('tooltip').style.display='none';};
}
function seedBands(runs){
  const groups=new Map(),window=Number($('smooth').value);
  runs.forEach(r=>{const c=r.config;const key=JSON.stringify(['algorithm','source','scenario','epsilon','gamma','lr','loss','training_episodes','forecast','clients','co','target_hours','width','batch_size','device','new_clients'].map(k=>c[k]));if(!groups.has(key))groups.set(key,[]);groups.get(key).push(r);});
  return [...groups.values()].filter(g=>new Set(g.map(r=>r.config.seed)).size>1).map(g=>{const seen=new Set();g=g.filter(r=>!seen.has(r.config.seed)&&seen.add(r.config.seed));const maps=g.map(r=>new Map(curves(r,'cost','cost',.001,window).map(p=>[p.x,p.y])));const data=[];for(const x of maps[0].keys()){const ys=maps.map(m=>m.get(x));if(ys.every(finite)){const m=avg(ys),sd=Math.sqrt(avg(ys.map(y=>(y-m)**2)));data.push({x,low:m-sd,high:m+sd});}}return {data};});
}
function render(){
 const active=state.details.find(r=>r.id===state.active)||state.details[0];if(!active)return;
 const window=Number($('smooth').value),events=active.events,train=events.filter(e=>e.type==='train'),last=train.at(-1),costs=events.filter(e=>e.type==='cost'),vals=events.filter(e=>e.type==='validation'),val=vals.at(-1),all=state.details;
 const days=last?.day||costs.at(-1)?.day||0,epochs=active.config.training_episodes||active.config.episodes||active.config.iterations*30;
 $('progress').textContent=`${fmt(epochs?days/epochs*100:0,1)}%`;$('progress-bar').style.width=`${Math.min(100,days/epochs*100)}%`;$('progress-caption').textContent=`${fmt(days,0)} / ${fmt(epochs,0)} training days`;
 $('validation').textContent=val?`$${fmt(val.daily*.001,3)}`:'—';$('validation-caption').textContent=val?`Best $${fmt(active.best_validation?.daily*.001,3)} / day · day 30–59`:'검증은 설정한 간격마다 진행됩니다';
 const cost=avg(costs.slice(-window).map(e=>meanField(e,'cost')));$('train-cost').textContent=finite(cost)?`$${fmt(cost*.001,3)}`:'—';$('train-caption').textContent=`Last ${Math.min(window,costs.length)} days · exploration ε=${active.config.epsilon}`;
 const violations=last?.clients?.reduce((s,c)=>s+(c.violations||0),0);$('violations').textContent=finite(violations)?fmt(violations,0):'—';$('violations').style.color=violations?'#b75247':'#295e45';
 let plateau='Early training',notes=[];const daily=costs.map(e=>meanField(e,'cost'));
 if(daily.length>=400){let now=daily.slice(-100),before=daily.slice(-200,-100),m=avg(now),change=(m-avg(before))/avg(before),cv=Math.sqrt(avg(now.map(v=>(v-m)**2)))/m;plateau=Math.abs(change)<.01&&cv<.05?'Plateau observed':'Still changing';notes.push(`최근 100일 비용 변화 ${fmt(change*100,1)}% · 변동계수 ${fmt(cv*100,1)}%`);}else notes.push('초기 학습 구간입니다. 짧은 실행으로 수렴을 판단하지 않습니다.');
 $('convergence').textContent=plateau;$('convergence').className='badge'+(plateau==='Plateau observed'?' warning':'');
 if(active.config.algorithm!=='MPC'&&last?.clients?.some(c=>(c.updates||0)===0))notes.push('Replay warm-up: 아직 optimizer update가 시작되지 않은 가구가 있습니다.');
 if(active.config.algorithm==='MPC')notes.push('MPC는 신경망 학습이 아닌 제어 시뮬레이션입니다. Loss·Q 지표는 생성하지 않습니다.');
 if(last?.clients?.some(c=>Math.abs(c.accounting_error||0)>.001))notes.push('가구별 DC 요금과 기록된 학습 비용의 불일치가 감지됐습니다.');
 if(active.config.evaluation_overlap_preserved)notes.push('Legacy 평가에는 학습 구간이 포함됩니다. 화면의 Validation은 별도 구간입니다.');
 $('notice').textContent=notes.join(' ');$('active-name').textContent=active.id;$('config').textContent=JSON.stringify(active.config,null,2);
 const compare=(type,key,scale=1,w=1)=>all.map((r,i)=>({name:r.id.split('/').at(-1),color:colors[i%colors.length],data:curves(r,type,key,scale,w)}));
 plot('cost-chart',compare('cost','cost',.001,window),{bands:$('axis').value==='day'?seedBands(all):[]});plot('val-chart',compare('validation','daily',.001));plot('loss-chart',compare('train','loss'),{log:$('log-loss').checked});
 const activeSeries=keys=>keys.map((key,i)=>({name:key,color:colors[i],data:curves(active,'train',key)}));
 plot('q-chart',activeSeries(['q_mean','target_mean']));plot('grad-chart',activeSeries(['grad_norm','target_grad_norm']));
 plot('composition-chart',['tou','dc'].map((key,i)=>({name:key.toUpperCase(),color:colors[i],data:curves(active,'cost',key,.001,window)})));
 plot('buffer-chart',activeSeries(['buffer_size','updates']));plot('resource-chart',[{name:'RSS MiB',color:colors[0],data:train.filter(e=>finite(e.rss_mb)).map(e=>({x:xs(e),y:e.rss_mb}))}]);
 $('throughput').textContent=`${fmt(last?.days_per_second,1)} days/s`;
 document.querySelector('.toolbar-note').textContent=`Live local telemetry · ${active.config.device||'cpu'} · ${active.config.threads||1} CPU thread(s)`;
 const client=Number($('client').value),clientIds=last?.clients?.map(c=>c.client)||[0];
 if($('client').options.length!==clientIds.length)$('client').innerHTML=clientIds.map(i=>`<option value="${i}">EMS ${i+1}</option>`).join('');
 const selected=clientIds.includes(client)?client:clientIds[0],schedule=last?.schedule?.[String(selected)]||[],row=last?.clients?.find(c=>c.client===selected);
 plot('schedule-chart',['battery','action','load','generation'].map((key,i)=>({name:key,color:colors[i],data:schedule.map(p=>({x:p.hour,y:p[key]*.1}))})));
 $('schedule-caption').textContent=`학습 day ${fmt(last?.day,0)} · 배터리 / 구매 / 수요 / 재생에너지 · kWh`;
 $('peak').textContent=row?`${fmt(row.peak*.1,1)} kWh`:'—';$('battery').textContent=row?`${fmt(row.final_battery*.1,1)} kWh`:'—';$('accounting').textContent=row?`$${fmt(row.accounting_error*.001,3)}`:'—';$('accounting').style.color=Math.abs(row?.accounting_error||0)>.001?'#b75247':'#295e45';
 $('eta').textContent=`${sec(last?.elapsed||active.status.seconds)} / ${sec(last?.eta_seconds)}`;$('resources').textContent=`CPU ${fmt(last?.cpu_percent,0)}% · ${fmt(last?.rss_mb,0)} MiB${finite(last?.gpu_util_percent)?' · GPU '+fmt(last.gpu_util_percent,0)+'%':''}`;
 $('footer-status').textContent=`Active: ${active.status.status} · ${active.config.versions?.torch||'unknown torch'}`;
}
function registry(){const runs=state.runs.filter(r=>(!$('algorithm').value||r.config.algorithm===$('algorithm').value)&&(!$('source').value||r.config.source===$('source').value));$('run-count').textContent=`${runs.length} experiments · ${runs.filter(r=>r.status.status==='running').length} running`;
 $('runs').innerHTML=runs.map(r=>{const d=r.latest?.day||0,n=r.config.training_episodes||r.config.episodes||r.config.iterations*30,status=r.status.status;return `<tr class="${r.id===state.active?'active':''}"><td><input type="checkbox" aria-label="Compare ${esc(r.id)}" data-id="${esc(r.id)}" ${state.selected.has(r.id)?'checked':''}></td><td><button data-active="${esc(r.id)}" title="${esc(r.id)}">${esc(r.id)}</button></td><td><span class="badge ${status==='failed'?'error':status==='running'?'warning':''}">${esc(status)}</span></td><td>${esc(r.config.algorithm)} / ${esc(r.config.source)}</td><td>${esc(r.config.seed)}</td><td>${fmt(d,0)} / ${fmt(n,0)}</td><td>${finite(r.best_validation?.daily)?'$'+fmt(r.best_validation.daily*.001,3):'—'}</td><td>${sec(r.status.seconds||r.latest?.elapsed)}</td></tr>`;}).join('');
 $('runs').querySelectorAll('input').forEach(el=>el.onchange=()=>{if(el.checked){if(state.selected.size>=6){el.checked=false;return;}state.selected.add(el.dataset.id);}else state.selected.delete(el.dataset.id);refresh();});$('runs').querySelectorAll('button').forEach(el=>el.onclick=()=>{state.active=el.dataset.active;state.selected.add(state.active);refresh();});
}
async function refresh(){if(state.paused||state.busy)return;state.busy=true;try{const response=await fetch('/api/runs');if(!response.ok)throw Error(response.status);state.runs=await response.json();if(!state.selected.size&&state.runs.length){state.active=state.runs[0].id;state.selected.add(state.active);}const valid=new Set(state.runs.map(r=>r.id));state.selected=new Set([...state.selected].filter(id=>valid.has(id)));state.details=await Promise.all([...state.selected].slice(0,6).map(async id=>{const response=await fetch('/api/run?id='+encodeURIComponent(id));if(!response.ok)throw Error(response.status);return response.json();}));registry();render();$('connection').textContent='● Live';$('connection').className='badge';$('updated').textContent='Updated '+new Date().toLocaleTimeString('ko-KR');}catch(error){$('connection').textContent='Disconnected';$('connection').className='badge error';$('updated').textContent=String(error);}finally{state.busy=false;}}
$('pause').onclick=()=>{state.paused=!state.paused;$('pause').textContent=state.paused?'▶ Resume live':'Ⅱ Pause live';$('connection').textContent=state.paused?'Paused':'● Live';if(!state.paused)refresh();};
['algorithm','source'].forEach(id=>$(id).onchange=registry);['smooth','axis','client','log-loss'].forEach(id=>$(id).onchange=render);window.addEventListener('resize',()=>render());refresh();setInterval(refresh,2000);
