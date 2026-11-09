"""Loopback-only live training dashboard. Uses stdlib HTTP; no cloud or extra services."""
import argparse
import csv
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlparse,parse_qs
from . import status as run_status

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
WEB=ROOT/'dashboard'
_cache={};_lock=threading.Lock()

def artifact(path,new,old):
    return path/new if (path/new).exists() else path/old

def read_json(path,default=None):
    try:return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return default

def read_events(path):
    with _lock:
        if not path.exists():return []
        size=path.stat().st_size
        item=_cache.setdefault(str(path),dict(offset=0,events=[]))
        if size<item['offset']:item.update(offset=0,events=[])
        with path.open('rb') as f:
            f.seek(item['offset'])
            while True:
                pos=f.tell();line=f.readline()
                if not line or not line.endswith(b'\n'):
                    item['offset']=pos;break
                try:item['events'].append(json.loads(line))
                except ValueError:pass
                item['offset']=f.tell()
        return list(item['events'])

def run_paths():
    return {p.parent.relative_to(RESULTS).as_posix():p.parent for p in RESULTS.rglob('config.json')}

MAX_COST_POINTS=2000

def compact_events(events):
    """Shrink a run's events for the browser: bin daily costs, keep the hourly schedule only on the last train event."""
    costs=[e for e in events if e.get('type')=='cost']
    other=[e for e in events if e.get('type')!='cost']
    trains=[i for i,e in enumerate(other) if e.get('type')=='train']
    other=[e if e.get('type')!='train' or i==trains[-1] else {k:v for k,v in e.items() if k!='schedule'}
           for i,e in enumerate(other)]
    if len(costs)>MAX_COST_POINTS:
        span=-(-len(costs)//MAX_COST_POINTS);binned=[]
        for start in range(0,len(costs),span):
            chunk=costs[start:start+span];clients={}
            for e in chunk:
                for c in e.get('clients',[]):
                    row=clients.setdefault(c['client'],{})
                    for k,v in c.items():
                        if k!='client' and isinstance(v,(int,float)):row.setdefault(k,[]).append(v)
            binned.append(dict(type='cost',day=chunk[-1]['day'],span=span,elapsed=chunk[-1].get('elapsed'),
                               clients=[dict(client=n,**{k:sum(v)/len(v) for k,v in row.items()}) for n,row in sorted(clients.items())]))
        costs=binned
    return other+costs

def live_status(path):
    return run_status.effective(read_json(path/'status.json',{'status':'unknown'}))

def planned_args(args):
    def value(flag,default=None):
        return args[args.index(flag)+1] if flag in args and args.index(flag)+1<len(args) else default
    episodes=value('--episodes')
    return dict(algorithm=value('--algorithm','TDAS'),source=value('--source','fixed'),seed=int(value('--seed',0)),
                scenario=int(value('--scenario',1)),training_episodes=int(episodes) if episodes else 30*int(value('--iterations',100)),
                args=args)

def queued_runs(existing):
    """Runs a suite has planned but not started; started runs are reported from their own folders."""
    rows=[];names={run_id.rsplit('/',1)[-1] for run_id in existing}
    for plan_path in RESULTS.rglob('plan.json'):
        statuses=read_json(plan_path.parent/'status.json',{}) or {}
        for name,entry in statuses.items():
            if not isinstance(entry,dict) or entry.get('status') not in ('queued','running'):continue
            run=(RESULTS/entry['run']) if entry.get('run') else None
            if run is not None and run.relative_to(RESULTS).as_posix() in existing:continue
            if name in names:continue  # started: its own folder reports the status (suite may record a bare name)
            rows.append(dict(id=f'queued:{name}',queued=True,suite=plan_path.parent.name,config=planned_args(entry.get('args',[])),
                             status=dict(status='queued',queued=entry.get('queued')),latest=None,best_validation=None,
                             modified=entry.get('queued',0)))
    return rows

def run_detail(run_id):
    paths=run_paths()
    if run_id not in paths:raise KeyError('Unknown run')
    path=paths[run_id];config=read_json(path/'config.json',{})
    events=read_events(artifact(path,'metrics/events.jsonl','metrics.jsonl'))
    if not events and (path/'cost_history.npy').exists():
        import numpy as np
        costs=np.load(path/'cost_history.npy',allow_pickle=False)
        if costs.ndim==1:costs=costs[None,:]
        events=[dict(type='cost',day=i+1,clients=[dict(client=j,cost=float(costs[j,i]))
                 for j in range(len(costs))]) for i in range(costs.shape[1])]
    return dict(id=run_id,config=config,status=live_status(path),events=compact_events(events),
                best_validation=read_json(artifact(path,'evaluation/best-validation.json','best-validation.json')))

def summaries():
    rows=[]
    paths=run_paths()
    for run_id,path in paths.items():
        events=read_events(artifact(path,'metrics/events.jsonl','metrics.jsonl'))
        latest=next((r for r in reversed(events) if r.get('type')=='train'),None)
        config=read_json(path/'config.json',{})
        rows.append(dict(id=run_id,config=config,status=live_status(path),
                         latest=latest,best_validation=read_json(artifact(path,'evaluation/best-validation.json','best-validation.json')),
                         modified=path.stat().st_mtime))
    rows+=queued_runs(set(paths))
    order={s:i for i,s in enumerate(('running','stalled','queued','interrupted','failed','stopped','completed'))}
    return sorted(rows,key=lambda r:(order.get(r['status']['status'],9),-r['modified']))

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,data,status=200,content_type='application/json; charset=utf-8'):
        body=data if isinstance(data,bytes) else json.dumps(data,allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        url=urlparse(self.path)
        try:
            if url.path=='/api/runs':return self.reply(summaries())
            if url.path=='/api/run':return self.reply(run_detail(parse_qs(url.query).get('id',[''])[0]))
            if url.path=='/api/health':return self.reply(dict(status='ok',service='energy-scheduling-monitor'))
            if url.path=='/api/diagnostics':
                data={}
                for name in ('toy_oracle_sensitivity','heldout_baselines'):
                    with (RESULTS/'diagnostics'/f'{name}.csv').open(encoding='utf-8',newline='') as f:
                        data[name]=list(csv.DictReader(f))
                return self.reply(data)
            files={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),
                   '/style.css':('style.css','text/css; charset=utf-8')}
            if url.path not in files:return self.reply(dict(error='Not found'),404)
            file,mime=files[url.path];return self.reply((WEB/file).read_bytes(),content_type=mime)
        except KeyError:return self.reply(dict(error='Unknown run'),404)
        except (ValueError,OSError) as exc:return self.reply(dict(error=str(exc)),500)

def main():
    global RESULTS
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--results',type=Path,default=RESULTS,help='Directory searched for run folders (default: results/)')
    args=parser.parse_args()
    RESULTS=args.results.resolve();server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Energy Scheduling Lab: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
