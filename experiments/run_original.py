"""Seeded, headless launcher for the actual repository scripts, not a new RL implementation.

Original evaluation and training/evaluation overlap are deliberately preserved.
This runner MUST NOT be used to claim held-out performance or hyperparameter optimality.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT='340e5556cef9e0065d7bf274639a94f917784631'
DATA_ORDER=['table1_load.npy','USA_CA_Oakland.Intl.AP.724930_TMY3_BASE.csv.npy',
            'USA_CA_Oxnard.AP.723927_TMY3_BASE.csv.npy','USA_CA_Salinas.Muni.AP.724917_TMY3_BASE.csv.npy',
            'USA_CA_San.Diego-Lindbergh.Field.722900_TMY3_BASE.csv.npy',
            'USA_CA_Santa.Barbara.Muni.AP.723925_TMY3_BASE.csv.npy',
            'USA_CA_Santa.Maria.Public.AP.723940_TMY3_BASE.csv.npy',
            'USA_CA_Visalia.Muni.AWOS.723896_TMY3_BASE.csv.npy']

# Named settings; explicit command-line flags always win. 'paper' reproduces the published code budget.
PRESETS={
    'tuned':dict(episodes=6000,target_hours=240,updates_per_day=24,cost_scale=100,width=512),
    'fast':dict(episodes=6000,target_hours=240,updates_per_day=12,cost_scale=100,width=256),
    'paper':dict(episodes=None,target_hours=20,updates_per_day=1,cost_scale=1,width=512),
}

def resolve_sites(patterns):
    """Comma-separated processed site names or glob patterns, e.g. 'tmy3/*_TX_*,korea/KOR_Seoul'."""
    names=[]
    for pattern in [p.strip() for p in patterns.split(',') if p.strip()]:
        matches=sorted(p.parent.relative_to(ROOT/'data'/'processed').as_posix() for p in (ROOT/'data'/'processed').glob(pattern+'/load.npy'))
        if not matches:raise SystemExit(f'No processed site matches {pattern!r}; run python -m experiments.datasets build')
        names+=[m for m in matches if m not in names]
    return names

def apply_preset(args):
    """Fill unset options from the preset; scenario-dependent paper values for epsilon and budget."""
    preset=args.preset or ('paper' if args.source=='upstream' else 'tuned')
    args.preset=preset
    toy=args.algorithm=='TDAS' and args.scenario in (1,2)
    for key,value in PRESETS[preset].items():
        if key=='episodes' and args.iterations is not None:
            continue  # an explicit --iterations wins over the preset's training-day budget
        if getattr(args,key) is None:
            setattr(args,key,value)
    if args.iterations is None:
        args.iterations=max(1,args.episodes//args.train_days) if args.episodes else 5000 if toy else 10000
    if args.epsilon is None:
        args.epsilon=0.01 if toy else 0.1
    if args.lr is None and preset=='paper' and args.algorithm=='Co-TDAS':
        args.lr=1e-5  # paper value; the Co-TDAS.py default (1e-4) is the tuned one
    return args

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm',choices=['TDAS','Co-TDAS'],default='TDAS')
    parser.add_argument('--source',choices=['upstream','fixed'],default='fixed')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--scenario',type=int,choices=[1,2,3,4],default=1)
    parser.add_argument('--preset',choices=sorted(PRESETS),help='tuned (default), fast, or paper (default for --source upstream)')
    parser.add_argument('--iterations',type=int,help='Passes over the training window (paper: 5000 toy / 10000)')
    parser.add_argument('--episodes',type=int,help='Exact number of training days (tuned/fast presets: 6000); overrides iterations')
    parser.add_argument('--epsilon',type=float,help='Default 0.01 for TDAS scenarios 1-2, otherwise 0.1 (paper)')
    parser.add_argument('--lr',type=float)
    parser.add_argument('--gamma',type=float,default=0.98)
    parser.add_argument('--forecast',type=int,choices=range(0,24),default=2)
    parser.add_argument('--clients',type=int,help='EMSs from the start (default 4, or all --sites)')
    parser.add_argument('--train-start',type=int,default=0,help='First day of the training window (0 = Jan 1)')
    parser.add_argument('--train-days',type=int,default=30,help='Length of the training window in days (paper: 30)')
    parser.add_argument('--sites',help='Processed households instead of the original data: comma-separated names or globs '
                        'under data/processed (e.g. "tmy3/*_TX_*"). TDAS uses the first; Co-TDAS one per EMS')
    parser.add_argument('--co',type=int,choices=[0,1],default=1)
    parser.add_argument('--new-clients',type=int,choices=[0,2],default=0)
    parser.add_argument('--log-every',type=int,default=25)
    parser.add_argument('--eval-every',type=int,default=500)
    parser.add_argument('--target-hours',type=int,help='Target sync period in hours; preset default')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--width',type=int,help='Hidden layer width; preset default')
    parser.add_argument('--loss',choices=['huber','mse'],default='huber')
    parser.add_argument('--frl-days',type=int,help='Co-TDAS: federated aggregation every this many days (paper: 1)')
    parser.add_argument('--clusters',help='Co-TDAS: comma-separated cluster label per EMS, e.g. 0,0,1,1; aggregate only within a label (default: all together)')
    parser.add_argument('--updates-per-day',type=int,help='Gradient steps per simulated day (paper: 1)')
    parser.add_argument('--cost-scale',type=float,help='Divide replayed costs by this value (paper: 1)')
    parser.add_argument('--start-size',type=int,help='Replay size before learning starts (paper code: 5000)')
    parser.add_argument('--buffer-limit',type=int,help='Replay capacity (paper code: 20000)')
    parser.add_argument('--epsilon-final',type=float,help='Linearly decay epsilon to this value')
    parser.add_argument('--epsilon-decay-days',type=int,help='Training days over which epsilon decays')
    parser.add_argument('--output',type=Path)
    args=apply_preset(parser.parse_args())
    if args.iterations<=0 or not 0<=args.epsilon<=1 or not 0<=args.gamma<1 or (args.lr is not None and args.lr<=0):
        parser.error('Require iterations>0, lr>0, epsilon in [0,1], gamma in [0,1)')
    if (args.episodes is not None and args.episodes<30) or args.log_every<1 or args.eval_every<0 or args.target_hours<1:
        parser.error('episodes>=30, log-every>=1, eval-every>=0 and target-hours>=1 required')
    if not(0<=args.train_start and 1<=args.train_days and args.train_start+args.train_days<=365):
        parser.error('Training window must lie inside the year: 0 <= train-start, train-start + train-days <= 365')
    sites=resolve_sites(args.sites) if args.sites else None
    if args.clients is None:
        args.clients=len(sites)-args.new_clients if sites and args.algorithm=='Co-TDAS' else 4
    available=len(sites) if sites else 8
    if args.algorithm=='Co-TDAS' and (args.clients<1 or args.clients+args.new_clients>available):
        parser.error(f'clients + new-clients must be between 1 and {available} available datasets')
    if args.clusters is not None:
        try:args.clusters=[int(x) for x in args.clusters.split(',')]
        except ValueError:parser.error('--clusters takes comma-separated integer labels')
        if args.algorithm!='Co-TDAS' or args.co!=1 or len(args.clusters)!=args.clients+args.new_clients:
            parser.error(f'--clusters needs Co-TDAS with --co 1 and one label per EMS ({args.clients+args.new_clients})')
    if sites and args.algorithm=='TDAS' and args.scenario not in (3,4):
        parser.error('--sites needs a real-data scenario (3 or 4); the seasonal tariff of scenarios 3-4 is used')
    if args.threads<1 or args.batch_size<1 or args.batch_size>5000 or args.width<1:
        parser.error('threads/width>=1 and batch-size in [1,5000] required')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work'/'matplotlib'))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    try:
        import numpy as np
        import torch
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        parser.exit(2,f'Missing dependency: {exc}. Install requirements.txt first.\n')
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA is unavailable in this environment')
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    from .core import validation_window
    from .artifacts import default_run_path,prepare,manifest,data_manifest
    output=prepare((args.output or default_run_path(ROOT,args.algorithm,args.source,args.scenario,args.seed)).resolve())
    import shutil
    support=output/'source'/'support';support.mkdir()
    for file in (ROOT/'experiments').glob('*.py'):shutil.copy2(file,support/file.name)
    source_path=ROOT/f'{args.algorithm}.py'
    if args.source=='upstream':
        source=subprocess.check_output(['git','show',f'{UPSTREAM_COMMIT}:{source_path.name}'],cwd=ROOT).decode('utf-8-sig')
    else:
        source=source_path.read_text(encoding='utf-8-sig')
    # One documented compatibility substitution, applied to both modes equally.
    source=source.replace('np.Inf','np.inf')
    source=source.replace("'log/' + directory", "str(_artifact_root / 'logs' / 'legacy')")
    overrides=dict(SCENARIO=args.scenario,iteration=args.iterations,epsilon=args.epsilon,
                   gamma=args.gamma,Tf=args.forecast,days=args.train_days,train_start=args.train_start,
                   test_days=30 if args.scenario in (1,2) else 365,
                   N=args.clients,N_new=args.clients,NEW=args.new_clients,CO=args.co,
                   freq=args.target_hours,
                   batch_size=args.batch_size,
                   load_file_list=sites or DATA_ORDER,generation_file_list=sites or ['generation.npy'])
    if sites and args.algorithm=='TDAS':
        # Scenario 3 reads load_file_list[0]; scenarios 3 and 4 share the seasonal tariff calendar.
        args.scenario=overrides['SCENARIO']=3
    if args.episodes is not None:
        overrides.update(epochs=args.episodes,iteration=max(1,args.episodes//args.train_days))
    if args.lr is not None:
        overrides['learning_rate']=args.lr
    for key in ('updates_per_day','cost_scale','start_size','buffer_limit','epsilon_final','epsilon_decay_days','frl_days','clusters'):
        if getattr(args,key) is not None:
            overrides[key]=getattr(args,key)
    class Settings(ast.NodeTransformer):
        def visit_Assign(self,node):
            # Only module-level assignments are fed to this transformer.
            names=[t.id for t in node.targets if isinstance(t,ast.Name)]
            key=next((n for n in names if n in overrides),None)
            if key:
                node.value=ast.parse(repr(overrides[key]),mode='eval').body
            return node
    tree=ast.parse(source)
    transformer=Settings()
    # Data-list assignments occur inside module-level conditional blocks too.
    def visit_top(node):
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            return node
        if isinstance(node,ast.Assign):
            return transformer.visit_Assign(node)
        for name in ('body','orelse'):
            if hasattr(node,name):
                setattr(node,name,[visit_top(child) for child in getattr(node,name)])
        return node
    tree.body=[visit_top(n) for n in tree.body]
    co=args.algorithm=='Co-TDAS'
    client='n' if co else '0'
    class Instrument(ast.NodeTransformer):
        def visit_ClassDef(self,node):
            node=self.generic_visit(node)
            if node.name=='Qnet':
                for call in ast.walk(node):
                    if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='Linear':
                        for arg in call.args:
                            if isinstance(arg,ast.Constant) and arg.value==512:arg.value=args.width
            if node.name in ('Qnet','ReplayBuffer'):
                helper='_model_on_device' if node.name=='Qnet' else '_replay_on_device'
                return [node,ast.parse(f'{node.name} = {helper}({node.name},_device)').body[0]]
            return node
        def visit_FunctionDef(self,node):
            if node.name=='train':
                node.body.append(ast.parse(f'_observer.loss({client},loss,q_a,target)').body[0])
                if args.loss=='mse':
                    for call in ast.walk(node):
                        if isinstance(call,ast.Attribute) and call.attr=='smooth_l1_loss':
                            call.attr='mse_loss'
                for call in ast.walk(node):
                    if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='torch' and call.func.attr in ('arange','tensor'):
                        if not any(k.arg=='device' for k in call.keywords):call.keywords.append(ast.keyword(arg='device',value=ast.Name(id='_device',ctx=ast.Load())))
            return node
        def visit_For(self,node):
            is_training=(isinstance(node.target,ast.Name) and node.target.id=='n_epi'
                         and 'epochs' in ast.unparse(node.iter))
            if is_training:
                class Transition(ast.NodeTransformer):
                    def visit_Expr(self,statement):
                        call=statement.value
                        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='put':
                            before='state[n]' if co else 'state'
                            hook=ast.parse(f'_observer.transition({client},n_epi,time,{before},state_prime,action,cost)').body[0]
                            return [statement,hook]
                        return statement
                node=Transition().visit(node)
                node.body.append(ast.parse('_observer.day(n_epi)').body[0])
                return node
            return self.generic_visit(node)
    tree=Instrument().visit(tree)
    ast.fix_missing_locations(tree)
    executed_source=ast.unparse(tree)
    (output/'source'/'executed_source.py').write_text(executed_source,encoding='utf-8')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    plt.show=lambda:plt.close('all')
    config=vars(args).copy();config['output']=str(output);config['sites']=sites
    config.update(training_days=args.train_days,evaluation_overlap_preserved=True,
                  upstream_commit=UPSTREAM_COMMIT,
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),load_file_order=overrides['load_file_list'],generation_file_order=overrides['generation_file_list'],
                  executed_source_sha256=hashlib.sha256(executed_source.encode()).hexdigest(),
                  training_episodes=overrides.get('epochs',args.train_days*args.iterations),pid=os.getpid(),
                  validation_interval=list(validation_window(args.train_start,args.train_days)),final_test_interval=[61,365],
                  timing_convention='fixed source: target in hours; upstream source: original days',
                  device=str(device),cuda_runtime=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                  artifact_schema=1,units=dict(energy_kwh=0.1,cost_usd=0.001),
                  versions={p:importlib.metadata.version(p) for p in ['numpy','torch','pandas','matplotlib','tqdm']})
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (output/'environment'/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    (output/'environment'/'data-manifest.json').write_text(json.dumps(data_manifest(ROOT),indent=2),encoding='utf-8')
    from .devices import model_on_device,replay_on_device
    namespace={'__name__':'__main__','__file__':str(source_path),'_device':device,
               '_model_on_device':model_on_device,'_replay_on_device':replay_on_device,'_artifact_root':output}
    from .telemetry import Observer
    observer=Observer(output,namespace,config,args.log_every,args.eval_every)
    namespace['_observer']=observer
    from . import status as run_status
    run_status.start(output,day=0,epochs=config['training_episodes'])
    run_status.install_stop_handlers()
    start=time.perf_counter()
    previous=os.getcwd()

    def save_outputs(prefix):
        for key in ('cost_history','result_cost_history','result_action_history','action_history'):
            if key in namespace:
                try:np.save(output/'history'/f'{key}.npy',np.asarray(namespace[key]))
                except ValueError:
                    for client,history in enumerate(namespace[key]):np.save(output/'history'/f'{key}-{client}.npy',np.asarray(history))
        models=namespace.get('q',[])
        for i,model in enumerate(models if isinstance(models,list) else [models]):
            torch.save(model.state_dict(),output/'checkpoints'/f'{prefix}-model-{i}.pt')
        optimizers=namespace.get('optimizers',[namespace.get('optimizer')])
        for i,optimizer in enumerate(optimizers):
            if optimizer is not None:torch.save(optimizer.state_dict(),output/'checkpoints'/f'{prefix}-optimizer-{i}.pt')

    def finish(state,**fields):
        run_status.write(output,state,seconds=time.perf_counter()-start,**fields)
        observer.finish(state,fields.get('error'))
        manifest(output,state,config)
    try:
        os.chdir(ROOT)
        # Upstream Co-TDAS writes log/; retain its behavior inside this worktree.
        with (output/'logs'/'stdout.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log):
            exec(compile(tree,str(source_path),'exec'),namespace)
        save_outputs('last')
        finish('completed')
    except (run_status.RunStopped,KeyboardInterrupt) as exc:
        # Partial models/histories are kept under a distinct name so they are never mistaken for finished ones.
        save_outputs('stopped')
        finish('stopped',reason=str(exc) or 'KeyboardInterrupt')
        print(f'Stopped: {output}',flush=True)
        raise SystemExit(130)
    except Exception as exc:
        finish('failed',error=repr(exc))
        raise
    finally:
        os.chdir(previous)
    print(f'Completed: {output}',flush=True)

if __name__=='__main__':
    main()
