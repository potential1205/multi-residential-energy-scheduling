import ast
import math
import random
from types import SimpleNamespace
import unittest
import numpy as np
from .core import Config, ROOT, feasible, step, state, tariff, toy_oracle, load_data

class SchedulingTests(unittest.TestCase):
    def test_mpc_discretization_respects_charging_rate_and_numpy_scalars(self):
        tree=ast.parse((ROOT/'MPC.py').read_text(encoding='utf-8-sig'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cont2disc')
        env=dict(np=np,action_set=np.arange(51),B_max=40,B_rate=20,C_max=20)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<mpc-discretization>','exec'),env)
        u=env['cont2disc'](np.array([[20.],[0.],[0.]]),np.array([[0.],[0.]]),[0],[4],np.array([1]))
        self.assertEqual(u[2,0],20)
        self.assertEqual(u[1,0],20)

    def test_optimized_action_selection_matches_original_mask(self):
        rng=np.random.default_rng(7)
        for name in ('TDAS.py','Co-TDAS.py'):
            tree=ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))
            qnet=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qnet')
            node=next(n for n in qnet.body if isinstance(n,ast.FunctionDef) and n.name=='sample_action')
            node.decorator_list=[]
            env=dict(math=math,random=random,battery_rate=20,battery_max=40,
                     torch=SimpleNamespace(argmin=np.argmin))
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<sample-action>','exec'),env)
            for _ in range(1000):
                battery=float(rng.integers(-40,81))/2
                rate=float(rng.integers(-60,61))/2
                values=rng.normal(size=21)
                allowed=[a for a in range(21) if 0<=battery+a<=40 and -20<=rate+a<=20]
                fake=SimpleNamespace(forward=lambda _:values)
                if not allowed:
                    with self.assertRaises(ValueError):
                        env['sample_action'](fake,None,0,battery,rate)
                    continue
                expected=min(allowed,key=lambda a:values[a])
                self.assertEqual(env['sample_action'](fake,None,0,battery,rate),expected)
                self.assertIn(env['sample_action'](fake,None,1,battery,rate),allowed)

    def test_new_clients_participate_in_aggregation(self):
        tree=ast.parse((ROOT/'Co-TDAS.py').read_text(encoding='utf-8-sig'))
        names={'get_grad','agg_grad','federated_learning'}
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        class Model:
            def __init__(self,value): self.value={'w':np.array([value],dtype=float)}
            def state_dict(self): return self.value
            def load_state_dict(self,value): self.value={k:v.copy() for k,v in value.items()}
        import copy
        env=dict(copy=copy,np=np,N=4,N_new=6,q_before=[Model(0) for _ in range(6)],
                 q=[Model(1) for _ in range(6)],q_global=Model(0),q_target=[Model(0) for _ in range(6)],
                 experience_counts=[48]*4+[24]*2)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<federated>', 'exec'),env)
        env['federated_learning']()
        self.assertTrue(all(np.isclose(m.value['w'][0],1) for m in env['q']))
        self.assertTrue(all(np.isclose(m.value['w'][0],1) for m in env['q_target']))
        weighted=env['agg_grad']({'w':np.array([0.])},[{'w':np.array([v])} for v in range(1,7)],[2]*4+[1]*2)
        self.assertAlmostEqual(weighted['w'][0],3.1)

    def test_optimizer_state_survives_training_calls(self):
        tree=ast.parse((ROOT/'Co-TDAS.py').read_text(encoding='utf-8-sig'))
        train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        self.assertNotIn('optim.Adam',ast.unparse(train))
        self.assertIn('optimizers[n]',ast.unparse(train))

    def test_all_admissible_actions_preserve_constraints(self):
        cfg=Config()
        for b in range(41):
            for load in (0,4,10,20):
                for gen in (0,4):
                    allowed,base=feasible(b,load,gen,cfg)
                    for a in allowed:
                        after=step(b,load,gen,a,cfg)
                        self.assertTrue(0<=after<=40)
                        self.assertTrue(abs(after-b)<=20)
                        self.assertEqual(after,base+a)

    def test_reject_infeasible_action(self):
        with self.assertRaises(ValueError):
            step(0,4,1,0,Config())

    def test_no_feasible_action_fails_explicitly(self):
        with self.assertRaises(ValueError):
            feasible(0,100,0,Config())

    def test_no_storage_has_analytic_optimum(self):
        for scenario in (1,2):
            for dc in (0,30,60):
                r=toy_oracle(Config(capacity=0,dc=dc),scenario,days=2)
                expected=2*(3*sum(tariff(0,scenario))+3*dc)
                self.assertEqual(r['total'],expected)
                self.assertTrue(all(a==3 for a in r['actions']))

    def test_storage_cannot_worsen_toy_optimum(self):
        costs=[toy_oracle(Config(capacity=c),1,days=2,initial=0)['total'] for c in (0,20,40)]
        self.assertTrue(all(a>=b for a,b in zip(costs,costs[1:])))

    def test_midnight_season_tariffs_match_actual_day(self):
        load=np.ones((365,24))
        obs=state(15,0,23,89,load,load,Config(),3)
        self.assertEqual(obs[26],tariff(89,3)[23])
        self.assertTrue(np.array_equal(obs[27:50],tariff(90,3)[:23]))

    def test_real_inputs_valid_and_explicit(self):
        with self.assertRaises(ValueError):
            load_data(3)
        for p in (ROOT/'data'/'load').glob('*.npy'):
            l,g,_=load_data(3,p.name)
            self.assertEqual(l.shape,g.shape)

    def test_original_co_cost_expression_uses_client_dc(self):
        tree=ast.parse((ROOT/'Co-TDAS.py').read_text(encoding='utf-8-sig'))
        expressions=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='cost' for t in n.targets)]
        self.assertEqual(len(expressions),1)
        expr=compile(ast.Expression(expressions[0]),'<co-cost>','eval')
        for n,dc in enumerate((20,25,30,35)):
            value=eval(expr,dict(action=4,TOUs=[tariff(0,1)]*4,n=n,time=23,
                                 DCs=[20,25,30,35],pD=30,actions=[4,8]))
            self.assertEqual(value,4*5+dc*8)

    def test_target_evaluation_is_no_grad(self):
        for name in ('TDAS.py','Co-TDAS.py'):
            tree=ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))
            train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
            self.assertTrue(any(isinstance(n,ast.With) and 'torch.no_grad()' in ast.unparse(n)
                                for n in train.body))

    def test_original_transition_matches_diagnostic_model(self):
        # Execute the repository's real pure transition function, without starting training.
        tree=ast.parse((ROOT/'TDAS.py').read_text(encoding='utf-8-sig'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='next_state')
        env=dict(np=np,feature=56,Tf=2,T=np.eye(24))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<transition>','exec'),env)
        load=np.full((30,24),4);gen=np.full((30,24),1)
        cfg=Config()
        for hour in (0,22,23):
            allowed,base=feasible(15,4,1,cfg)
            a=int(allowed[len(allowed)//2])
            result=env['next_state'](0,hour,base,a,[a],tariff(0,1),load,gen,30)
            self.assertEqual(result[0],step(15,4,1,a,cfg))
            self.assertEqual(result[1],0 if hour==23 else a)

if __name__=='__main__':
    unittest.main()
