"""Actual-data baselines, DC sensitivity and exact toy DP; not paper reproduction."""
import csv
import hashlib
import importlib.util
import json
import platform
import numpy as np
from .core import ROOT, Config, load_data, evaluate, toy_oracle

def main():
    out = ROOT/'results'/'diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sorted((ROOT/'data').rglob('*.npy')):
        a = np.load(path, allow_pickle=False)
        manifest.append(dict(file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             shape=list(a.shape),dtype=str(a.dtype),min=int(a.min()),max=int(a.max()),mean=float(a.mean())))
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows = []
    for path in sorted((ROOT/'data'/'load').glob('*.npy')):
        load, generation, name = load_data(3,path.name)
        for dc in (20,25,30,35):
            cfg = Config(dc=dc)
            policies = {
                'minimum_purchase':lambda obs,allowed:int(allowed[0]),
                'cheap_hour_fill':lambda obs,allowed:int(allowed[-1] if obs[26]==5 else allowed[0]),
            }
            for policy,fn in policies.items():
                rows.append(dict(dataset=name,policy=policy,dc_rate=dc,
                                 **evaluate(fn,load,generation,cfg,3)))
    with (out/'heldout_baselines.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    oracle_rows=[]
    for scenario in (1,2):
        for cap in (0,20,40,80):
            for dc in (0,20,30,60):
                result=toy_oracle(Config(capacity=cap,dc=dc),scenario,days=30)
                result.pop('actions')
                oracle_rows.append(dict(scenario=scenario,capacity=cap,dc_rate=dc,**result))
    with (out/'toy_oracle_sensitivity.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(oracle_rows[0]));writer.writeheader();writer.writerows(oracle_rows)
    environment=dict(python=platform.python_version(),numpy=np.__version__,
                     dependencies={m:bool(importlib.util.find_spec(m)) for m in ('torch','pandas','matplotlib','casadi','do_mpc')},
                     test_interval='days [60,365)',training_run=False,paper_reproduced=False,
                     notes=['Deterministic baselines; no seed confidence intervals apply.',
                            'Toy DP uses L=4/G=1, cyclic terminal inventory, original discrete TDAS dynamics.',
                            'Real-data heuristics ignore future observations. No tuning on final test data.'])
    (out/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    print(json.dumps(dict(data_files=len(manifest),heldout_baseline_runs=len(rows),toy_exact_runs=len(oracle_rows),
                         output=str(out),training_run=False),indent=2),flush=True)

if __name__=='__main__':
    main()
