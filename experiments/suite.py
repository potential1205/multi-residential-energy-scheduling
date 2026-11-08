"""Bounded parallel experiment plan runner; every child is a separate immutable run."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from .core import ROOT
from . import status as run_status

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    plan=json.loads(args.plan.read_text(encoding='utf-8'));suite=ROOT/'results'/'suites'/plan['name']
    suite.mkdir(parents=True,exist_ok=False);(suite/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    # Every planned run is visible from the start; the dashboard shows these as 'queued'.
    statuses={r['name']:dict(status='queued',run=f"runs/{r['name']}",args=r['args'],queued=time.time()) for r in plan['runs']}
    lock=threading.Lock()
    def save():(suite/'status.json').write_text(json.dumps(statuses,indent=2),encoding='utf-8')
    save()
    def update(name,value):
        with lock:
            statuses[name]={**statuses[name],**value}
            temporary=suite/'status.next.json';temporary.write_text(json.dumps(statuses,indent=2),encoding='utf-8')
            temporary.replace(suite/'status.json')
    def run(spec):
        update(spec['name'],dict(status='running',started=time.time()))
        destination=ROOT/'results'/'runs'/spec['name']
        command=[sys.executable,'-m','experiments.run_original',*spec['args'],'--output',str(destination)]
        env=os.environ.copy();env.setdefault('OMP_NUM_THREADS','1');env.setdefault('MKL_NUM_THREADS','1')
        with (suite/f"{spec['name']}.launcher.log").open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        # The child records why it ended (completed / stopped / failed); fall back to the exit code.
        child=run_status.read(destination).get('status')
        final=child if child in run_status.FINAL else 'completed' if result.returncode==0 else 'failed'
        return dict(status=final,returncode=result.returncode,finished=time.time())
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futures={pool.submit(run,spec):spec for spec in plan['runs']}
        for future in as_completed(futures):
            spec=futures[future]
            try:result=future.result()
            except Exception as exc:result=dict(status='failed',error=repr(exc))
            update(spec['name'],result);print(spec['name'],result['status'],flush=True)
    if any(s['status']!='completed' for s in statuses.values()):raise SystemExit(1)

if __name__=='__main__':main()
