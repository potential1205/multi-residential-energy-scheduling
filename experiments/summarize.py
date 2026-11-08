"""Small, shareable summaries; never selects hyperparameters using test metrics."""
import csv
import json
from pathlib import Path
import numpy as np
from .core import ROOT

def main():
    rows=[]
    for config_path in sorted((ROOT/'results'/'runs').glob('*/config.json')):
        path=config_path.parent;config=json.loads(config_path.read_text(encoding='utf-8'))
        status=json.loads((path/'status.json').read_text())
        if status['status']!='completed':continue
        best=json.loads((path/'evaluation'/'best-validation.json').read_text()) if (path/'evaluation'/'best-validation.json').exists() else None
        records=[json.loads(line) for line in (path/'metrics'/'events.jsonl').read_text().splitlines()]
        validation=[r for r in records if r['type']=='validation'];train=[r for r in records if r['type']=='train']
        test=json.loads((path/'evaluation'/'final-test.json').read_text()) if (path/'evaluation'/'final-test.json').exists() else None
        controller=json.loads((path/'evaluation'/'controller.json').read_text()) if (path/'evaluation'/'controller.json').exists() else None
        rows.append(dict(run=path.name,algorithm=config['algorithm'],source=config['source'],scenario=config['scenario'],
                         seed=config['seed'],episodes=config['training_episodes'],device=config['device'],threads=config.get('threads',1),
                         width=config.get('width',512) if config['algorithm']!='MPC' else None,batch_size=config.get('batch_size',32) if config['algorithm']!='MPC' else None,lr=config.get('resolved',{}).get('learning_rate',config.get('lr')),
                         seconds=status.get('seconds'),best_validation_usd=best['daily']*.001 if best else None,
                         final_validation_usd=validation[-1]['daily']*.001 if validation else None,
                         test_daily_usd=test['mean_daily_usd'] if test else None,
                         horizon=config.get('horizon'),forecast_noise=config.get('noise'),
                         controller_daily_usd=controller['total_usd']/controller['days'] if controller else None,
                         controller_violations=controller['violations'] if controller else None,
                         violations=sum(c.get('violations',0) for c in train[-1]['clients']) if train else None))
    out=ROOT/'results'/'summaries';out.mkdir(exist_ok=True)
    (out/'runs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if rows:
        with (out/'runs.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
