"""Read-only training observers: JSONL metrics without changing RNG or optimizer state."""
from collections import defaultdict
from contextlib import contextmanager
import ctypes
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np

def resident_mb():
    if os.name != 'nt':
        return None
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[
            (k,ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    handle=ctypes.windll.kernel32.GetCurrentProcess
    handle.restype=ctypes.c_void_p
    query=ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    return counters.WorkingSetSize/2**20 if query(handle(),ctypes.byref(counters),counters.cb) else None

@contextmanager
def preserve_rng(torch):
    py=random.getstate();np_state=np.random.get_state();torch_state=torch.random.get_rng_state()
    cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py);np.random.set_state(np_state);torch.random.set_rng_state(torch_state)
        if cuda_state is not None:torch.cuda.set_rng_state_all(cuda_state)

class Observer:
    def __init__(self,output,namespace,config,log_every=25,eval_every=500):
        import torch
        self.torch=torch;self.output=Path(output);self.ns=namespace;self.config=config
        self.log_every=log_every;self.eval_every=eval_every
        self.start=time.perf_counter();self.cpu_start=time.process_time()
        self.losses=defaultdict(list);self.transitions=defaultdict(list)
        self.updates=defaultdict(int);self.violations=defaultdict(int)
        self.best_validation=math.inf
        self.validation_cohort=None
        self.gpu_sample={};self.gpu_sample_time=-math.inf
        self.stream=(self.output/'metrics'/'events.jsonl').open('w',encoding='utf-8',buffering=1)

    def emit(self,item):
        item.update(timestamp=time.time(),elapsed=time.perf_counter()-self.start)
        # Fail loudly on NaN/Infinity rather than silently writing misleading charts.
        self.stream.write(json.dumps(item,allow_nan=False)+'\n');self.stream.flush()

    def loss(self,client,loss,q_a,target):
        self.updates[client]+=1
        if self.updates[client]%10 != 0:
            return
        with self.torch.no_grad():
            residual=(target-q_a).detach()
            values=dict(loss=float(loss.item()),td_abs=float(residual.abs().mean().item()),
                        td_std=float(residual.std(unbiased=False).item()),
                        q_mean=float(q_a.detach().mean().item()),target_mean=float(target.mean().item()))
            co=self.config['algorithm']=='Co-TDAS'
            for key,model in [('grad_norm',self.ns['q'][client] if co else self.ns['q']),
                              ('target_grad_norm',self.ns['q_target'][client] if co else self.ns['q_target'])]:
                norms=[self.torch.linalg.vector_norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]
                values[key]=float(self.torch.linalg.vector_norm(self.torch.stack(norms)).item()) if norms else 0.
        if not all(math.isfinite(v) for v in values.values()):
            raise FloatingPointError(f'Non-finite training diagnostics: {values}')
        self.losses[client].append(values)

    def transition(self,client,day,hour,before,after,action,cost):
        cfg=self.ns
        delta=float(after[0]-before[0])
        invalid=(after[0]<-1e-6 or after[0]>cfg['battery_max']+1e-6 or abs(delta)>cfg['battery_rate']+1e-6)
        self.violations[client]+=int(invalid)
        tf=cfg['Tf']
        self.transitions[client].append(dict(hour=int(hour),action=int(action),battery=float(before[0]),
                                            next_battery=float(after[0]),load=float(before[50]),
                                            generation=float(before[51+tf]),price=float(before[26]),cost=float(cost)))

    def validate(self,day):
        from .core import Config,evaluate,load_data,validation_window
        torch=self.torch;ns=self.ns;co=self.config['algorithm']=='Co-TDAS'
        models=ns['q'][:ns.get('N_new',1)] if co else [ns['q']]
        if self.validation_cohort!=len(models):
            self.best_validation=math.inf;self.validation_cohort=len(models)
        rows=[];window=validation_window(ns.get('train_start',0),ns['days'])
        with preserve_rng(torch),torch.no_grad():
            for client,model in enumerate(models[:ns.get('N_new',1)]):
                old_mode=model.training;model.eval()
                dc=ns['DCs'][client] if co else ns['pD']
                cfg=Config(dc=dc,forecast=ns['Tf'])
                if co:
                    load,gen,_=load_data(3,ns['load_file_list'][client])
                    scenario=1 if np.array_equal(ns['TOUs'][client],ns['s']) else 2
                else:
                    scenario=ns['SCENARIO']
                    file=None if scenario<3 else ns['load_file_list'][0 if scenario==3 else 2]
                    load,gen,_=load_data(scenario,file)
                def policy(obs,allowed):
                    if co: obs=np.append(obs,dc).astype(np.float32)
                    q=model(torch.from_numpy(obs))
                    return int(allowed[int(q[allowed].argmin().item())])
                result=evaluate(policy,load,gen,cfg,scenario,start=window[0],stop=window[1])
                model.train(old_mode)
                rows.append(dict(client=client,**result))
            average=float(np.mean([r['daily'] for r in rows]))
            if average<self.best_validation:
                self.best_validation=average
                for i,model in enumerate(models):
                    torch.save(model.state_dict(),self.output/'checkpoints'/f'best-validation-{i}.pt')
                (self.output/'evaluation'/'best-validation.json').write_text(json.dumps(dict(day=day,daily=average,clients=rows)),encoding='utf-8')
        self.emit(dict(type='validation',day=day,daily=average,best_daily=self.best_validation,clients=rows,
                       interval=list(window),selection_metric='mean daily validation cost; lower is better'))

    def day(self,day):
        ns=self.ns;co=self.config['algorithm']=='Co-TDAS';completed=day+1
        if completed==1:
            self.config['resolved']={k:ns[k] for k in ('learning_rate','gamma','epsilon','batch_size','buffer_limit','start_size',
                 'Tf','feature','battery_max','battery_rate','freq','epochs','days','pD','DCs','TOUs','CO','N','NEW',
                 'updates_per_day','cost_scale','epsilon_final','epsilon_decay_days','train_start','frl_days','clusters') if k in ns}
            (self.output/'config.json').write_text(json.dumps(self.config,indent=2),encoding='utf-8')
        clients=[]
        for client,trace in self.transitions.items():
            actions=[t['action'] for t in trace]
            dc=ns['DCs'][client] if co else ns['pD']
            peak=max(actions) if actions else 0
            tou=sum(t['action']*t['price'] for t in trace)
            actual=sum(t['cost'] for t in trace)
            losses=self.losses[client]
            memory=ns['memory'][client] if co else ns['memory']
            row=dict(client=client,cost=actual,tou=tou,dc=dc*peak,canonical_cost=tou+dc*peak,
                     accounting_error=actual-(tou+dc*peak),peak=peak,
                     final_battery=trace[-1]['next_battery'] if trace else None,
                     buffer_size=memory.size(),updates=self.updates[client],violations=self.violations[client],
                     **({k:float(np.mean([v[k] for v in losses[-10:]])) for k in losses[-1]} if losses else {}))
            clients.append(row)
        if completed%self.log_every==0 or completed==ns['epochs'] or completed==1:
            elapsed=time.perf_counter()-self.start
            if self.config.get('device')=='cuda' and elapsed-self.gpu_sample_time>=5:
                try:
                    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
                    values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
                                                    '--format=csv,noheader,nounits'],text=True,timeout=2,creationflags=flags).splitlines()[0].split(',')
                    self.gpu_sample={k:float(v.strip()) for k,v in zip(('gpu_util_percent','gpu_memory_util_percent','gpu_total_used_mb','gpu_power_w'),values)}
                except (OSError,ValueError,subprocess.SubprocessError):self.gpu_sample={}
                self.gpu_sample_time=elapsed
            from .status import write as write_status
            write_status(self.output,'running',day=completed,epochs=ns['epochs'])
            # The logger is called after all active clients have advanced one day.
            per_client_steps=completed*24
            self.emit(dict(type='train',day=completed,epochs=ns['epochs'],step=per_client_steps,
                           clients=clients,eta_seconds=elapsed/max(completed,1)*(ns['epochs']-completed),
                           days_per_second=completed/max(elapsed,1e-9),
                           cpu_percent=100*(time.process_time()-self.cpu_start)/max(elapsed,1e-9),rss_mb=resident_mb(),
                           gpu_allocated_mb=self.torch.cuda.memory_allocated()/2**20 if self.config.get('device')=='cuda' else None,
                           gpu_reserved_mb=self.torch.cuda.memory_reserved()/2**20 if self.config.get('device')=='cuda' else None,
                           **self.gpu_sample,
                           schedule={str(k):v for k,v in self.transitions.items()}))
        # Keep all daily costs cheaply for faithful smoothing and seed summaries.
        self.emit(dict(type='cost',day=completed,clients=[dict(client=r['client'],cost=r['cost'],
                       tou=r['tou'],dc=r['dc'],canonical_cost=r['canonical_cost'],peak=r['peak']) for r in clients]))
        self.transitions.clear()
        if self.eval_every and (completed%self.eval_every==0 or completed==ns['epochs']):
            self.validate(completed)

    def finish(self,status,error=None):
        if self.stream.closed:return
        self.emit(dict(type='status',status=status,error=error))
        self.stream.close()
