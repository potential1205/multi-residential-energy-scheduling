"""Independent greedy evaluation of a saved validation-selected checkpoint."""
import argparse
import ast
import json
from pathlib import Path
import random
import math
import numpy as np
import torch
from torch import nn
from .core import ROOT,Config,evaluate,load_data
from .artifacts import manifest

def evaluate_run(path,start=61,stop=365):
    torch.set_num_threads(1);path=Path(path)
    config=json.loads((path/'config.json').read_text(encoding='utf-8'))
    source=(path/'source'/'executed_source.py').read_text(encoding='utf-8')
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qnet')
    co=config['algorithm']=='Co-TDAS';feature=52+2*config['forecast']+int(co)
    env=dict(torch=torch,nn=nn,feature=feature,math=math,random=random,battery_max=40,battery_rate=20)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<saved-qnet>','exec'),env)
    rows=[];dcs=config.get('resolved',{}).get('DCs',[20,25,30,35,20,25,30,35])
    clients=config['clients']+config.get('new_clients',0) if co else 1
    with torch.no_grad():
        for client in range(clients):
            model=env['Qnet']();checkpoint=path/'checkpoints'/f'best-validation-{client}.pt'
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
            dc=dcs[client] if co else 30
            cfg=Config(dc=dc,forecast=config['forecast'])
            scenario=(1 if client%2==0 else 2) if co else config['scenario']
            file=config['load_file_order'][client] if co else None if scenario<3 else config['load_file_order'][0 if scenario==3 else 2]
            load,gen,_=load_data(3 if co else scenario,file)
            def policy(obs,allowed):
                if co:obs=np.append(obs,dc).astype(np.float32)
                values=model(torch.from_numpy(obs));return int(allowed[values[allowed].argmin().item()])
            result=evaluate(policy,load,gen,cfg,scenario,start=start,stop=stop)
            rows.append(dict(client=client,**result))
    result=dict(protocol='independent greedy evaluation; validation-selected checkpoint',interval=[start,stop],
                initial_battery=15,clients=rows,mean_daily_usd=float(np.mean([r['daily'] for r in rows]))*.001,
                notes=['Future observations through Tf are known, as assumed by the paper.',
                       'The last forecast window wraps to the start of the 365-day series, following legacy periodic data convention.'])
    (path/'evaluation'/'final-test.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    manifest(path,'completed',config)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    parser.add_argument('--start',type=int,default=61);parser.add_argument('--stop',type=int,default=365)
    args=parser.parse_args();print(json.dumps(evaluate_run(args.run,args.start,args.stop),indent=2))

if __name__=='__main__':main()
