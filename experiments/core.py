"""NumPy diagnostics. Units and transitions follow TDAS.py, including its clipping order.

These are experimental checks, not a claim to reproduce the published figures.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMER = np.array([5,5,5,5,5,5,5,10,10,10,10,15,15,15,15,15,15,10,10,5,5,5,5,5])
WINTER = np.array([5,5,5,5,5,5,5,15,15,15,25,10,10,10,10,10,10,15,15,5,5,5,5,5])

def tariff(day, scenario):
    if scenario == 1:
        return SUMMER
    if scenario == 2:
        return WINTER
    return SUMMER if 90 <= day < 273 else WINTER

@dataclass(frozen=True)
class Config:
    capacity: int = 40
    rate: int = 20
    max_grid: int = 20
    dc: float = 30
    forecast: int = 2

def feasible(battery, load, generation, cfg):
    base = min(battery - load + generation, cfg.capacity)
    actions = np.arange(cfg.max_grid + 1)
    valid = ((base + actions >= 0) & (base + actions <= cfg.capacity)
             & (np.abs(base - battery + actions) <= cfg.rate))
    if not valid.any():
        raise ValueError(f'No feasible grid purchase: B={battery}, L={load}, G={generation}')
    return actions[valid], base

def step(battery, load, generation, action, cfg):
    actions, base = feasible(battery, load, generation, cfg)
    if action not in actions:
        raise ValueError(f'Infeasible action {action}')
    return float(base + action)

def state(battery, peak, hour, day, load, generation, cfg, scenario, co=False):
    length = len(load)
    offsets = np.arange(cfg.forecast + 1) + hour
    indices = (day + offsets // 24) % length
    hours = offsets % 24
    # Each future tariff uses its own calendar day, including season boundaries.
    future = np.array([tariff(day + (hour+i)//24, scenario)[(hour+i)%24] for i in range(24)])
    parts = [np.array([battery, peak]), np.eye(24)[hour], future,
             load[indices, hours], generation[indices, hours]]
    if co:
        parts.append(np.array([cfg.dc]))
    return np.concatenate(parts).astype(np.float32)

def load_data(scenario, file=None):
    if scenario in (1, 2):
        return np.full((365,24),4), np.full((365,24),1), 'toy'
    # Explicit names avoid os.listdir-dependent scenario identity.
    if file is None:
        raise ValueError('Pass an explicit load filename; original scenario mapping depends on os.listdir order')
    name = file
    load = np.load(ROOT/'data'/'load'/name, allow_pickle=False)
    generation = np.load(ROOT/'data'/'generation'/'generation.npy', allow_pickle=False)
    for data in (load, generation):
        if data.shape != (365,24) or not np.isfinite(data).all() or np.any(data < 0):
            raise ValueError('Expected finite nonnegative data with shape (365,24)')
    return load, generation, name

def evaluate(policy, load, generation, cfg, scenario, start=60, stop=365, initial=15, trace=False):
    """Greedy rollout. With trace=True, also return per-day TOU/DC costs and hourly actions/battery."""
    battery, tou_cost, dc_cost, peaks = min(initial,cfg.capacity), 0., 0., []
    days = []
    for day in range(start, stop):
        peak, day_tou, actions, levels = 0, 0., [], []
        for hour in range(24):
            allowed, base = feasible(battery, load[day,hour], generation[day,hour], cfg)
            obs = state(battery, peak, hour, day, load, generation, cfg, scenario)
            action = int(policy(obs, allowed))
            levels.append(float(battery))
            battery = step(battery, load[day,hour], generation[day,hour], action, cfg)
            peak = max(peak, action)
            day_tou += action * tariff(day,scenario)[hour]
            actions.append(action)
        tou_cost += day_tou
        dc_cost += cfg.dc * peak
        peaks.append(peak)
        if trace:
            days.append(dict(day=day, tou=day_tou, dc=cfg.dc*peak, peak=peak, actions=actions, battery=levels))
    result = dict(total=tou_cost+dc_cost, tou=tou_cost, dc=dc_cost,
                  daily=(tou_cost+dc_cost)/(stop-start), mean_peak=float(np.mean(peaks)),
                  final_battery=battery, days=stop-start, violations=0)
    if trace:
        result['trace'] = days
    return result

def toy_oracle(cfg, scenario=1, days=30, initial=15):
    """Exact finite-horizon integer DP with cyclic terminal battery = initial.

    State is (battery, within-day peak); actions and transitions match TDAS.
    Cyclic inventory prevents a one-time benefit from consuming initial energy.
    """
    initial = min(initial,cfg.capacity)
    b = np.arange(cfg.capacity+1)[:,None,None]
    p = np.arange(cfg.max_grid+1)[None,:,None]
    a = np.arange(cfg.max_grid+1)[None,None,:]
    base = b - 3  # toy L=4, G=1
    next_b = np.clip(base+a,0,cfg.capacity)
    valid = (base+a >= 0) & (base+a <= cfg.capacity) & (np.abs(a-3) <= cfg.rate)
    next_p = np.maximum(p,a)
    value = np.full((cfg.capacity+1,cfg.max_grid+1),np.inf)
    value[initial,:] = 0
    policies = []
    for t in reversed(range(days*24)):
        hour = t%24
        future_peak = np.zeros_like(next_p) if hour==23 else next_p
        q = a * tariff(t//24,scenario)[hour] + value[next_b,future_peak]
        if hour==23:
            q = q + cfg.dc * next_p
        q = np.where(valid,q,np.inf)
        policies.append(q.argmin(axis=2).astype(np.uint8))
        value = q.min(axis=2)
    policies.reverse()
    battery, peak, tou, dc = initial, 0, 0., 0.
    trace = []
    for t, policy in enumerate(policies):
        hour = t%24
        action = int(policy[int(battery),peak])
        battery = step(battery,4,1,action,cfg)
        peak = max(peak,action)
        tou += action*tariff(t//24,scenario)[hour]
        if hour == 23:
            dc += cfg.dc*peak
            peak=0
        trace.append(action)
    if battery != initial or not np.isclose(tou+dc,value[initial,0]):
        raise AssertionError('DP reconstruction failed')
    return dict(total=tou+dc,tou=tou,dc=dc,daily=(tou+dc)/days,
                final_battery=battery,days=days,actions=trace)
