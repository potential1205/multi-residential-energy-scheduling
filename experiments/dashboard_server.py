"""Loopback-only live training dashboard. Uses stdlib HTTP; no cloud or extra services."""
import argparse
import csv
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlparse,parse_qs

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
    return dict(id=run_id,config=config,status=read_json(path/'status.json',{'status':'unknown'}),events=events,
                best_validation=read_json(artifact(path,'evaluation/best-validation.json','best-validation.json')))

def summaries():
    rows=[]
    for run_id,path in run_paths().items():
        events=read_events(artifact(path,'metrics/events.jsonl','metrics.jsonl'))
        latest=next((r for r in reversed(events) if r.get('type')=='train'),None)
        config=read_json(path/'config.json',{})
        rows.append(dict(id=run_id,config=config,status=read_json(path/'status.json',{'status':'unknown'}),
                         latest=latest,best_validation=read_json(artifact(path,'evaluation/best-validation.json','best-validation.json')),
                         modified=path.stat().st_mtime))
    return sorted(rows,key=lambda r:r['modified'],reverse=True)

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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args();server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Energy Scheduling Lab: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
