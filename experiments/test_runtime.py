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

if __name__=='__main__':unittest.main()
