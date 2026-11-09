"""Runtime checks with actual PyTorch, plus artifact and telemetry safety tests."""
import ast
import collections
import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from torch import nn,optim
from torch.nn import functional as F
from .core import ROOT
from .devices import model_on_device,replay_on_device
from .telemetry import preserve_rng
from .artifacts import prepare,manifest
from .dashboard_server import read_events,run_detail

def definitions(name):
    tree=ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    env=dict(torch=torch,nn=nn,optim=optim,F=F,np=np,math=math,collections=collections,
             random=random,copy=copy,feature=56,buffer_limit=100,batch_size=4,Tf=2,
             gamma=.98,battery_max=40,battery_rate=20,learning_rate=1e-4)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),name,'exec'),env)
    return env

class RuntimeTests(unittest.TestCase):
    def test_real_training_step_detaches_target_and_preserves_fractional_cost(self):
        torch.set_num_threads(1)
        for name in ('TDAS.py','Co-TDAS.py'):
            env=definitions(name);co=name.startswith('Co-')
            memory=env['ReplayBuffer']();state=np.zeros(56,dtype=np.float32)
            state[0]=15;state[50]=4;state[53]=1
            for _ in range(8):memory.put((state.copy(),3,1.75,state.copy()))
            self.assertTrue(torch.all(memory.sample(4)[2]==1.75))
            q=env['Qnet']();target=env['Qnet']();target.load_state_dict(q.state_dict())
            optimizer=optim.Adam(q.parameters(),lr=1e-4)
            env.update(q=[q] if co else q,q_target=[target] if co else target,
                       memory=[memory] if co else memory,optimizers=[optimizer],optimizer=optimizer)
            train=env['train'];train(0) if co else train();train(0) if co else train()
            self.assertTrue(all(p.grad is None for p in target.parameters()))
            self.assertTrue(all(torch.isfinite(p).all() for p in q.parameters()))
            self.assertTrue(all(int(v['step'].item())==2 for v in optimizer.state.values()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_forward_and_replay_match_cpu(self):
        env=definitions('TDAS.py');cls=env['Qnet'];cpu=cls()
        expected=cpu(torch.randn(3,56))
        x=torch.randn(3,56);expected=cpu(x).detach()
        gpu_cls=model_on_device(cls,torch.device('cuda'));gpu=gpu_cls();gpu.load_state_dict(cpu.state_dict())
        torch.testing.assert_close(gpu(x).cpu(),expected,rtol=1e-4,atol=1e-5)
        buffer_cls=replay_on_device(env['ReplayBuffer'],torch.device('cuda'));buffer=buffer_cls()
        for _ in range(4):buffer.put((np.zeros(56),0,1.25,np.zeros(56)))
        self.assertTrue(all(v.device.type=='cuda' for v in buffer.sample(4)))

    def test_rng_observation_has_no_effect_on_training_stream(self):
        random.seed(1);np.random.seed(1);torch.manual_seed(1)
        py=random.getstate();ns=np.random.get_state();ts=torch.random.get_rng_state().clone()
        with preserve_rng(torch):random.random();np.random.rand();torch.rand(10)
        self.assertEqual(random.getstate(),py);self.assertTrue(np.array_equal(np.random.get_state()[1],ns[1]))
        self.assertTrue(torch.equal(torch.random.get_rng_state(),ts))

    def test_artifacts_are_immutable_and_checksummed(self):
        with tempfile.TemporaryDirectory() as temp:
            path=prepare(Path(temp)/'run');(path/'config.json').write_text('{}')
            (path/'metrics'/'events.jsonl').write_text('{"type":"status"}\n')
            with self.assertRaises(FileExistsError):prepare(path)
            manifest(path,'completed',{})
            record=json.loads((path/'manifest.json').read_text())
            self.assertEqual(record['status'],'completed');self.assertTrue(all(len(f['sha256'])==64 for f in record['files']))

    def test_reader_handles_partial_live_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'events.jsonl';path.write_bytes(b'{"x":1}\n{"x":')
            self.assertEqual(read_events(path),[{'x':1}])
            with path.open('ab') as f:f.write(b'2}\n')
            self.assertEqual(read_events(path),[{'x':1},{'x':2}])
        with self.assertRaises(KeyError):run_detail('../.venv')

    def test_status_lifecycle_detects_dead_and_silent_runs(self):
        import os,socket,subprocess,sys,time
        from . import status
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);status.start(path,day=0,epochs=10)
            self.assertEqual(status.effective(status.read(path))['status'],'running')
            status.write(path,'running',day=5)
            record=status.read(path);self.assertEqual((record['day'],record['epochs'],record['pid']),(5,10,os.getpid()))
            self.assertEqual(status.effective(record,now=time.time()+status.STALL_SECONDS+1)['status'],'stalled')
            child=subprocess.Popen([sys.executable,'-c','pass']);child.wait()
            status.write(path,'running',pid=child.pid,host=socket.gethostname())
            self.assertEqual(status.effective(status.read(path))['status'],'interrupted')
            status.write(path,'stopped',reason='SIGTERM')
            self.assertEqual(status.effective(status.read(path))['status'],'stopped')
            with self.assertRaises(ValueError):status.write(path,'done')

    def test_clusters_option_is_validated(self):
        import subprocess,sys
        for extra in (['--clusters','0,1'],['--clusters','0,0,1,x'],['--clusters','0,0,1,1','--co','0']):
            result=subprocess.run([sys.executable,'-m','experiments.run_original','--algorithm','Co-TDAS','--clients','4',*extra],
                                  cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,2,extra)
            self.assertIn('--clusters',result.stderr)

    def test_dashboard_lists_queued_runs_once(self):
        from . import dashboard_server as server
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'_suites'/'p').mkdir(parents=True);(root/'_suites'/'p'/'plan.json').write_text('{}')
            (root/'_suites'/'p'/'status.json').write_text(json.dumps({
                'a':dict(status='queued',run='a',args=['--scenario','3','--episodes','600','--seed','2']),
                'b':dict(status='running',run='b',args=[]),'c':dict(status='completed',run='c',args=[]),
                'd':dict(status='running',run='d',args=[])}))  # suite recorded a bare name; folder is runs/d
            (root/'b').mkdir();(root/'b'/'config.json').write_text('{}')
            old=server.RESULTS;server.RESULTS=root
            try:rows=server.queued_runs({'b','runs/d'})
            finally:server.RESULTS=old
        self.assertEqual([r['id'] for r in rows],['queued:a'])
        self.assertEqual((rows[0]['config']['seed'],rows[0]['config']['training_episodes']),(2,600))

    def test_dashboard_bins_long_cost_histories(self):
        from .dashboard_server import compact_events,MAX_COST_POINTS
        events=[dict(type='cost',day=d+1,clients=[dict(client=0,cost=float(d))]) for d in range(5000)]
        events+=[dict(type='train',day=10,schedule={'0':[1]}),dict(type='train',day=20,schedule={'0':[2]})]
        out=compact_events(events);costs=[e for e in out if e['type']=='cost'];trains=[e for e in out if e['type']=='train']
        self.assertLessEqual(len(costs),MAX_COST_POINTS);self.assertEqual(costs[0]['span'],3)
        self.assertEqual(costs[0]['clients'][0]['cost'],1.0);self.assertEqual(costs[-1]['day'],5000)
        self.assertNotIn('schedule',trains[0]);self.assertEqual(trains[1]['schedule'],{'0':[2]})

if __name__=='__main__':unittest.main()
