"""Run the repository's actual do-mpc controller with scalar API compatibility fixes."""
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .core import ROOT
from .artifacts import default_run_path,prepare,manifest,data_manifest
from .run_original import DATA_ORDER,UPSTREAM_COMMIT

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scenario',type=int,choices=[1,2,3,4],default=1)
    parser.add_argument('--horizon',type=int,choices=[6,24,30],default=24);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--source',choices=['fixed','upstream'],default='fixed');parser.add_argument('--noise',type=float,default=0)
    parser.add_argument('--start-day',type=int,default=0);parser.add_argument('--stop-day',type=int)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work'/'matplotlib'));np.random.seed(args.seed)
    if args.stop_day is not None and not(0<=args.start_day<args.stop_day<=365):parser.error('Require 0<=start<stop<=365')
    output=prepare((args.output or default_run_path(ROOT,'MPC',args.source,args.scenario,args.seed)).resolve())
    source=(ROOT/'MPC.py').read_text(encoding='utf-8-sig') if args.source=='fixed' else subprocess.check_output(['git','show',f'{UPSTREAM_COMMIT}:MPC.py'],cwd=ROOT).decode('utf-8-sig')
    # Compatibility edits apply equally; original upstream constraints remain untouched.
    source=source.replace('int(t_now)','int(np.asarray(t_now).item())')
    if args.source=='upstream':source=source.replace('def cont2disc(u0, x0, L, G, sigma):','def cont2disc(u0, x0, L, G, sigma):\n    L, G, sigma = (float(np.asarray(v).item()) for v in (L, G, sigma))')
    source=source.replace('for k in trange(total_iter):','for k in trange(total_iter_nopadding):')
    source=source.replace('90 <= day < 273','90 <= day + _start_day < 273')
    tree=ast.parse(source)
    overrides=dict(SCENARIO=args.scenario,n_horizon=args.horizon,noise_max=args.noise*100,
                   load_file_list=DATA_ORDER,generation_file_list=['generation.npy'])
    if args.stop_day is not None:overrides.update(days=args.stop_day-args.start_day,test_days=args.stop_day-args.start_day)
    class Calendar(ast.NodeTransformer):
        def visit_Assign(self,node):
            names=[t.id for t in node.targets if isinstance(t,ast.Name)]
            for name in names:
                if name in overrides:node.value=ast.parse(repr(overrides[name]),mode='eval').body
            return self.generic_visit(node)
        def visit_Subscript(self,node):
            if args.stop_day is not None and args.scenario>=3:
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='load':
                    call=ast.unparse(node.value)
                    return ast.parse(f'np.concatenate(({call}[_start_day:_start_day+days],{call}[:days_padding]))',mode='eval').body
                if isinstance(node.slice,ast.BinOp) and isinstance(node.slice.op,ast.Mod) and ast.unparse(node.slice)=='day % days':
                    node.slice=ast.Name(id='day',ctx=ast.Load())
            return self.generic_visit(node)
    tree=Calendar().visit(tree)
    for node in tree.body:
        if isinstance(node,ast.Assign):
            names=[t.id for t in node.targets if isinstance(t,ast.Name)]
            for name in names:
                if name in overrides:node.value=ast.parse(repr(overrides[name]),mode='eval').body
        if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='k':
            node.body.append(ast.parse('_observe(k,x0,u0)').body[0])
    ast.fix_missing_locations(tree);executed=ast.unparse(tree)
    (output/'source'/'executed_source.py').write_text(executed,encoding='utf-8')
    config=vars(args).copy();config['output']=str(output);config.update(algorithm='MPC',training_episodes=overrides.get('days',30 if args.scenario<3 else 365),
        device='cpu',upstream_commit=UPSTREAM_COMMIT,artifact_schema=1,
        executed_source_sha256=hashlib.sha256(executed.encode()).hexdigest(),
        versions={p:importlib.metadata.version(p) for p in ['numpy','casadi','do-mpc','scipy']},
        perfect_forecast=args.noise==0,evaluation_overlap_preserved=False,load_file_order=DATA_ORDER,
        protocol='MPC controller evaluation, not neural training; realized states independently checked')
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (output/'environment'/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    (output/'environment'/'data-manifest.json').write_text(json.dumps(data_manifest(ROOT),indent=2),encoding='utf-8')
    start=time.perf_counter();namespace={'__name__':'__main__','_start_day':args.start_day};trace=[];violations=0;previous_battery=15
    stream=(output/'metrics'/'events.jsonl').open('w',encoding='utf-8',buffering=1)
    def observe(k,x,u):
        nonlocal violations,previous_battery
        b=float(x[0,0])/10;a=float(u[0,0])/10
        invalid=not(-1e-5<=b<=40+1e-5 and -1e-5<=a<=20+1e-5 and abs(b-previous_battery)<=20+1e-5)
        violations+=int(invalid)
        trace.append(dict(hour=k%24,action=a,battery=previous_battery,next_battery=b,
                          load=float(np.asarray(namespace['L_data'][k]).item())/10,
                          generation=float(np.asarray(namespace['G_data'][k]).item())/10,price=namespace['p_tou_'][k]))
        previous_battery=b
        if k%24==23:
            day=k//24+1;recent=trace[-24:];tou=sum(t['action']*t['price'] for t in recent);peak=max(t['action'] for t in recent)
            row=dict(client=0,cost=tou+30*peak,tou=tou,dc=30*peak,canonical_cost=tou+30*peak,
                     accounting_error=0,peak=peak,final_battery=b,violations=violations)
            common=dict(day=day,elapsed=time.perf_counter()-start,timestamp=time.time())
            for event in (dict(type='cost',clients=[row]),dict(type='train',epochs=namespace['days'],clients=[row],
                schedule={'0':recent},eta_seconds=(time.perf_counter()-start)/day*(namespace['days']-day),days_per_second=day/max(time.perf_counter()-start,1e-9))):
                stream.write(json.dumps(dict(**event,**common))+'\n')
            run_status.write(output,'running',day=day,epochs=namespace['days'])
    namespace['_observe']=observe
    from . import status as run_status
    run_status.start(output,day=0,epochs=namespace.get('days'));run_status.install_stop_handlers()
    previous=os.getcwd()
    try:
        os.chdir(ROOT)
        with (output/'logs'/'stdout.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):exec(compile(tree,'MPC.py','exec'),namespace)
        np.savez_compressed(output/'history'/'realized.npz',**{k:np.array([r[k] for r in trace]) for k in trace[0]})
        summary=dict(violations=violations,days=namespace['days'],horizon=args.horizon,noise=args.noise,
                     total_usd=sum(t['action']*t['price'] for t in trace)*.001+sum(30*max(t['action'] for t in trace[d:d+24]) for d in range(0,len(trace),24))*.001)
        (output/'evaluation'/'controller.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        run_status.write(output,'completed',seconds=time.perf_counter()-start)
    except (run_status.RunStopped,KeyboardInterrupt) as exc:
        run_status.write(output,'stopped',seconds=time.perf_counter()-start,reason=str(exc) or 'KeyboardInterrupt');raise SystemExit(130)
    except Exception as exc:
        run_status.write(output,'failed',seconds=time.perf_counter()-start,error=repr(exc));raise
    finally:
        stream.close();os.chdir(previous);manifest(output,json.loads((output/'status.json').read_text())['status'],config)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
