"""Static analysis reports for finished runs: figures, convergence statistics and monthly costs.

Single run:   python -m experiments.report results/runs/<run>
Seed group:   python -m experiments.report --group cool-default results/runs/base-s3-seed*
Compare:      python -m experiments.report --compare --out fig.png "paper=results/runs/base-s3-*" "tuned=results/runs/A-s3-*"

Costs are reported in USD (internal unit x 0.001). Convergence uses only the validation
interval (days 30-59) that the training data never contains; the yearly rollout is reported
for comparison with the paper's Table IV and is never used for selection.
"""
import argparse
import ast
import calendar
import json
import math
import os
from pathlib import Path
import random
import numpy as np
from .core import ROOT, Config, evaluate, load_data, tariff

USD = 0.001
MONTH_DAYS = [calendar.monthrange(2017, m)[1] for m in range(1, 13)]
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
# Monthly cumulative total cost ($) from Table IV of the paper (TSG 2023).
PAPER_TABLE_IV = {
    3: {'Greedy': [31.53,29.89,23.85,20.61,17.81,22.17,36.21,42.25,25.44,22.61,25.33,30.23],
        'MPC-24': [19.72,18.81,14.21,13.76,11.76,15.08,25.33,29.43,17.52,13.81,15.82,19.27],
        'TDAS (paper)': [19.69,18.73,14.14,14.17,12.30,15.38,27.29,32.16,18.10,14.02,15.58,19.04]},
    4: {'Greedy': [36.06,32.33,26.49,22.30,20.18,19.42,20.84,19.15,22.17,26.89,29.71,35.66],
        'MPC-24': [23.41,20.86,17.02,16.13,14.72,14.20,14.99,14.09,15.99,17.76,19.34,22.88],
        'TDAS (paper)': [23.60,21.13,17.66,17.07,15.83,15.48,16.23,15.38,16.84,17.95,22.87,23.87]},
}
TOY_OPTIMUM_DAILY_USD = 16.20 / 30  # Table III, both toy scenarios


def events(path):
    out = {'train': [], 'cost': [], 'validation': [], 'status': []}
    for line in (Path(path)/'metrics'/'events.jsonl').read_text(encoding='utf-8').splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        out.setdefault(record['type'], []).append(record)
    return out


def rolling(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or window <= 1:
        return values
    kernel = np.ones(min(window, len(values)))
    sums = np.convolve(values, kernel, mode='full')[:len(values)]
    counts = np.convolve(np.ones(len(values)), kernel, mode='full')[:len(values)]
    return sums / counts


def convergence(days, values, tol, window=3):
    """First evaluation day whose trailing `window`-eval mean is within tol of the final plateau.

    The plateau is the median of the last 20% of evaluations (at least 3). DQN validation keeps
    oscillating by a few percent, so "stays inside the band forever" is not a usable criterion.
    """
    if len(values) < 3:
        return None
    values = np.asarray(values, dtype=float)
    plateau = float(np.median(values[-max(3, len(values)//5):]))
    smooth = rolling(values, window)
    hit = np.nonzero(smooth <= plateau * (1 + tol))[0]
    return int(days[hit[0]]) if len(hit) else None


def first_reach(days, values, target):
    hit = [d for d, v in zip(days, values) if v <= target]
    return int(hit[0]) if hit else None


def scenario_data(config, client=0):
    order = config['load_file_order']
    if config['algorithm'] == 'Co-TDAS':
        scenario = 1 if client % 2 == 0 else 2
        load, gen, _ = load_data(3, order[client])
        return scenario, load, gen
    scenario = config['scenario']
    file = None if scenario < 3 else order[0 if scenario == 3 else 2]
    load, gen, _ = load_data(scenario, file)
    return scenario, load, gen


def greedy_policy(cfg):
    """Paper's Greedy baseline: buy max(L-G, 0) each hour (nearest admissible amount)."""
    def policy(obs, allowed):
        need = max(obs[50] - obs[51 + cfg.forecast], 0)
        return int(allowed[np.argmin(np.abs(allowed - need))])
    return policy


def load_model(path, config, client):
    import torch
    from torch import nn
    source = (path/'source'/'executed_source.py').read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Qnet')
    co = config['algorithm'] == 'Co-TDAS'
    env = dict(torch=torch, nn=nn, math=math, random=random, battery_max=40, battery_rate=20,
               feature=52 + 2*config['forecast'] + int(co))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<saved-qnet>', 'exec'), env)
    checkpoint = path/'checkpoints'/f'best-validation-{client}.pt'
    if not checkpoint.exists():
        checkpoint = path/'checkpoints'/f'last-model-{client}.pt'
    model = env['Qnet']()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    return model, checkpoint.name


def year_rollout(path, config, client=0):
    import torch
    torch.set_num_threads(1)
    co = config['algorithm'] == 'Co-TDAS'
    dc = config.get('resolved', {}).get('DCs', [30]*8)[client] if co else 30
    cfg = Config(dc=dc, forecast=config['forecast'])
    scenario, load, gen = scenario_data(config, client)
    model, checkpoint = load_model(path, config, client)

    def policy(obs, allowed):
        if co:
            obs = np.append(obs, dc)
        with torch.no_grad():
            q = model(torch.from_numpy(obs.astype(np.float32)))
        return int(allowed[int(q[allowed].argmin().item())])
    stop = 365 if scenario >= 3 else 30
    learned = evaluate(policy, load, gen, cfg, scenario, start=0, stop=stop, trace=True)
    greedy = evaluate(greedy_policy(cfg), load, gen, cfg, scenario, start=0, stop=stop, trace=True)
    return checkpoint, learned, greedy


def monthly(trace):
    totals, start = [], 0
    for days in MONTH_DAYS:
        part = trace[start:start + days]
        totals.append(sum(d['tou'] + d['dc'] for d in part) * USD)
        start += days
    return totals


def analyse(path):
    path = Path(path)
    config = json.loads((path/'config.json').read_text(encoding='utf-8'))
    ev = events(path)
    co = config['algorithm'] == 'Co-TDAS'
    clients = sorted({c['client'] for e in ev['cost'] for c in e['clients']})
    days = np.array([e['day'] for e in ev['cost']])
    daily = {k: np.array([[next((c[k] for c in e['clients'] if c['client'] == n), np.nan)
                            for e in ev['cost']] for n in clients]) * (USD if k != 'peak' else 1)
             for k in ('cost', 'tou', 'dc', 'peak')}
    val_days = [e['day'] for e in ev['validation']]
    val = [e['daily'] * USD for e in ev['validation']]
    scenario, load, gen = scenario_data(config)
    cfg = Config(dc=config.get('resolved', {}).get('DCs', [30])[0] if co else 30, forecast=config['forecast'])
    greedy_val = evaluate(greedy_policy(cfg), load, gen, cfg, scenario, start=30, stop=60)['daily'] * USD
    reference = TOY_OPTIMUM_DAILY_USD if scenario in (1, 2) and not co else None
    summary = dict(run=path.name, algorithm=config['algorithm'], scenario=config['scenario'], seed=config['seed'],
                   training_days=config['training_episodes'], resolved=config.get('resolved', {}),
                   seconds=json.loads((path/'status.json').read_text()).get('seconds'),
                   updates=[c.get('updates') for c in (ev['train'][-1]['clients'] if ev['train'] else [])],
                   validation=dict(days=val_days, daily_usd=val,
                                   best_usd=min(val) if val else None,
                                   best_day=val_days[int(np.argmin(val))] if val else None,
                                   last_usd=val[-1] if val else None,
                                   greedy_usd=greedy_val, reference_usd=reference,
                                   plateau_usd=float(np.median(val[-max(3, len(val)//5):])) if len(val) >= 3 else None,
                                   converged_day_5pct=convergence(val_days, val, .05),
                                   converged_day_2pct=convergence(val_days, val, .02),
                                   reached_greedy_minus_20pct=first_reach(val_days, val, .8*greedy_val),
                                   reached_reference_plus_2pct=first_reach(val_days, val, 1.02*reference) if reference else None),
                   train=dict(final_30day_usd=float(np.nanmean(daily['cost'][:, -30:])) if len(days) else None,
                              final_30day_peak=float(np.nanmean(daily['peak'][:, -30:])) if len(days) else None))
    year = None
    if (path/'checkpoints').exists() and any((path/'checkpoints').glob('*-0.pt')):
        checkpoint, learned, greedy = year_rollout(path, config)
        year = dict(checkpoint=checkpoint, daily_usd=learned['daily']*USD, tou_usd=learned['tou']*USD,
                    dc_usd=learned['dc']*USD, greedy_daily_usd=greedy['daily']*USD,
                    monthly_usd=monthly(learned['trace']) if len(learned['trace']) == 365 else None,
                    first_day=learned['trace'][0])
        summary['year'] = year
    (path/'evaluation').mkdir(exist_ok=True)
    (path/'evaluation'/'report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    figure(path, config, ev, days, daily, summary)
    return summary


def figure(path, config, ev, days, daily, summary):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'work'/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
    fig.suptitle(f"{path.name}  ·  {config['algorithm']}  scenario {config['scenario']}  seed {config['seed']}", fontsize=13)
    window = max(1, min(300, len(days)//50))
    a = ax[0, 0]
    for n in range(daily['cost'].shape[0]):
        a.plot(days, rolling(daily['cost'][n], window), lw=1.2, label=f'EMS {n+1}' if daily['cost'].shape[0] > 1 else 'total')
    if daily['cost'].shape[0] == 1:
        a.plot(days, rolling(daily['tou'][0], window), lw=1, ls='--', label='TOU part')
        a.plot(days, rolling(daily['dc'][0], window), lw=1, ls=':', label='DC part')
    a.set(title=f'Training cost (rolling {window} days, includes exploration)', xlabel='training day', ylabel='USD / day')
    a.legend(fontsize=8)
    a = ax[0, 1]
    v = summary['validation']
    if v['days']:
        a.plot(v['days'], v['daily_usd'], 'o-', ms=3, label='TDAS greedy policy')
        a.axhline(v['greedy_usd'], color='gray', ls='--', label=f"Greedy {v['greedy_usd']:.3f}")
        if v['reference_usd']:
            a.axhline(v['reference_usd'], color='green', ls=':', label=f"optimum {v['reference_usd']:.3f}")
        for key, color in (('converged_day_5pct', 'orange'), ('converged_day_2pct', 'red')):
            if v[key]:
                a.axvline(v[key], color=color, alpha=.5, label=f"within {key[-4]}% of plateau @ {v[key]}")
    a.set(title='Validation cost (days 30-59, never trained on)', xlabel='training day', ylabel='USD / day')
    a.legend(fontsize=8)
    train = ev['train']
    tdays = [e['day'] for e in train]
    def field(key):
        rows = [[c[key] for c in e['clients'] if c.get(key) is not None] for e in train]
        return [float(np.mean(r)) if r else np.nan for r in rows]
    a = ax[0, 2]
    if train and 'loss' in train[-1]['clients'][0]:
        a.semilogy(tdays, field('loss'), label='loss')
        a.semilogy(tdays, field('td_abs'), label='|TD error|')
        a.semilogy(tdays, field('grad_norm'), label='grad norm', alpha=.6)
    a.set(title='Optimisation (scaled training units)', xlabel='training day')
    a.legend(fontsize=8)
    a = ax[1, 0]
    if train and 'q_mean' in train[-1]['clients'][0]:
        a.plot(tdays, field('q_mean'), label='Q(s,a) mean')
        a.plot(tdays, field('target_mean'), label='Bellman target mean')
    a.set(title='Value estimates (scaled training units)', xlabel='training day')
    a.legend(fontsize=8)
    a = ax[1, 1]
    year = summary.get('year')
    if year and year['monthly_usd'] and config['scenario'] in PAPER_TABLE_IV:
        x = np.arange(12)
        a.bar(x - .3, year['monthly_usd'], .3, label='this run')
        for i, (name, values) in enumerate(PAPER_TABLE_IV[config['scenario']].items()):
            a.plot(x, values, marker='os^'[i], lw=1, label=name)
        a.set_xticks(x, MONTHS)
        a.set(title=f"Monthly cost over the year (best-validation model, {year['daily_usd']*365:.1f} $/yr)", ylabel='USD / month')
        a.legend(fontsize=8)
    else:
        a.plot(days, rolling(daily['peak'].mean(axis=0), window))
        a.set(title='Daily peak grid purchase (rolling)', xlabel='training day', ylabel='0.1 kWh')
    a = ax[1, 2]
    if year:
        d = year['first_day']
        scenario, load, gen = scenario_data(config)
        hours = np.arange(24)
        a.step(hours, np.array(d['actions'])*.1, where='post', lw=2, label='grid purchase')
        a.plot(hours, np.array(d['battery'])*.1, lw=2, label='battery (start of hour)')
        a.plot(hours, load[0]*.1, 'k-', lw=1, label='load')
        a.plot(hours, gen[0]*.1, 'k--', lw=1, label='generation')
        b = a.twinx()
        b.fill_between(hours, np.array(tariff(0, scenario))*.01, step='post', color='gray', alpha=.15)
        b.set_ylabel('TOU $/kWh')
        a.set(title=f"Schedule on day 0 (peak {d['peak']*.1:.1f} kWh)", xlabel='hour', ylabel='kWh')
        a.legend(fontsize=8, loc='upper left')
    fig.savefig(path/'figures'/'report.png', dpi=110)
    plt.close(fig)


def group(name, paths, out=None):
    rows = [analyse(p) for p in paths]
    out = Path(out or ROOT/'results'/'reports'/name)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'work'/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    curves = [r['validation'] for r in rows if r['validation']['days']]
    if curves:
        length = min(len(c['days']) for c in curves)
        d = np.array(curves[0]['days'][:length])
        y = np.array([c['daily_usd'][:length] for c in curves])
        ax[0].plot(d, y.T, color='C0', alpha=.25, lw=1)
        ax[0].plot(d, y.mean(0), color='C0', lw=2, label=f'mean of {len(y)} seeds')
        ax[0].fill_between(d, y.mean(0)-y.std(0), y.mean(0)+y.std(0), color='C0', alpha=.15)
        ax[0].axhline(curves[0]['greedy_usd'], color='gray', ls='--', label='Greedy')
        if curves[0]['reference_usd']:
            ax[0].axhline(curves[0]['reference_usd'], color='green', ls=':', label='optimum')
        ax[0].set(title=f'{name}: validation cost', xlabel='training day', ylabel='USD / day')
        ax[0].legend()
    years = [r['year'] for r in rows if r.get('year') and r['year']['monthly_usd']]
    scenario = rows[0]['scenario']
    if years and scenario in PAPER_TABLE_IV:
        x = np.arange(12)
        m = np.array([y['monthly_usd'] for y in years])
        ax[1].bar(x, m.mean(0), yerr=m.std(0), alpha=.6, label='this config (mean ± std)')
        for i, (label, values) in enumerate(PAPER_TABLE_IV[scenario].items()):
            ax[1].plot(x, values, marker='os^'[i], label=label)
        ax[1].set_xticks(x, MONTHS)
        ax[1].set(title='Monthly cost vs paper Table IV', ylabel='USD / month')
        ax[1].legend()
    fig.savefig(out/'validation.png', dpi=110)
    plt.close(fig)

    def stat(key, source='validation'):
        values = [r[source][key] for r in rows if r.get(source) and r[source].get(key) is not None]
        return (float(np.mean(values)), float(np.std(values)), len(values)) if values else None
    summary = dict(name=name, runs=[r['run'] for r in rows],
                   best_validation_usd=stat('best_usd'), plateau_validation_usd=stat('plateau_usd'),
                   converged_day_5pct=stat('converged_day_5pct'), converged_day_2pct=stat('converged_day_2pct'),
                   year_daily_usd=stat('daily_usd', 'year'), greedy_validation_usd=rows[0]['validation']['greedy_usd'],
                   seconds=[r['seconds'] for r in rows])
    (out/'summary.json').write_text(json.dumps(dict(summary=summary, runs=rows), indent=2), encoding='utf-8')
    lines = [f'# {name}', '', 'Convergence day = first validation point (3-eval mean) within 5% / 2% of the final plateau '
             '(median of the last 20% of evaluations).', '',
             '| run | best val $/day | @day | plateau $/day | within 5% day | within 2% day | year $/day | seconds |',
             '|---|---|---|---|---|---|---|---|']
    for r in rows:
        v, y = r['validation'], r.get('year') or {}
        lines.append(f"| {r['run']} | {v['best_usd']:.3f} | {v['best_day']} | {v['plateau_usd'] or float('nan'):.3f} | {v['converged_day_5pct']} | "
                     f"{v['converged_day_2pct']} | {y.get('daily_usd', float('nan')):.3f} | {r['seconds'] or 0:.0f} |")
    fmt = lambda s: '—' if s is None else f'{s[0]:.3f} ± {s[1]:.3f}' if s[0] < 100 else f'{s[0]:.0f} ± {s[1]:.0f}'
    lines += ['', f"- best validation: {fmt(summary['best_validation_usd'])} $/day (Greedy {summary['greedy_validation_usd']:.3f})",
              f"- plateau validation: {fmt(summary['plateau_validation_usd'])} $/day",
              f"- within 5% of plateau: day {fmt(summary['converged_day_5pct'])}; within 2%: day {fmt(summary['converged_day_2pct'])}",
              f"- year rollout: {fmt(summary['year_daily_usd'])} $/day", '', '![validation](validation.png)']
    (out/'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def compare(groups, out, title='Validation cost by configuration'):
    """Overlay seed-mean validation curves of several configurations (log-scaled training days)."""
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'work'/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    reference = None
    for i, (label, paths) in enumerate(groups.items()):
        curves = []
        for path in paths:
            ev = events(path)['validation']
            curves.append(([e['day'] for e in ev], [e['daily'] * USD for e in ev]))
        length = min(len(c[0]) for c in curves)
        days = np.array(curves[0][0][:length])
        y = np.array([c[1][:length] for c in curves])
        ax.plot(days, y.mean(0), lw=2, color=f'C{i}', label=f'{label} (n={len(y)})')
        ax.fill_between(days, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=f'C{i}', alpha=.15)
        if reference is None:
            config = json.loads((Path(paths[0])/'config.json').read_text(encoding='utf-8'))
            scenario, load, gen = scenario_data(config)
            co = config['algorithm'] == 'Co-TDAS'
            cfg = Config(dc=config.get('resolved', {}).get('DCs', [30])[0] if co else 30, forecast=config['forecast'])
            reference = evaluate(greedy_policy(cfg), load, gen, cfg, scenario, start=30, stop=60)['daily'] * USD
            if scenario in (1, 2) and config['algorithm'] == 'TDAS':
                ax.axhline(TOY_OPTIMUM_DAILY_USD, color='green', ls=':', label='optimum')
    ax.axhline(reference, color='gray', ls='--', label='Greedy')
    ax.set_xscale('log')
    ax.set(title=title, xlabel='training day (log scale)', ylabel='validation cost, USD / day (days 30-59)')
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--group', help='Aggregate the runs (e.g. seeds) under results/reports/<name>')
    parser.add_argument('--out', type=Path, help='Directory for the group report, or PNG path for --compare')
    parser.add_argument('--title', default='Validation cost by configuration', help='Figure title for --compare')
    parser.add_argument('--compare', action='store_true',
                        help='Treat each RUNS item as label=glob and overlay the configurations')
    args = parser.parse_args()
    if args.compare:
        import glob
        groups = {}
        for item in map(str, args.runs):
            label, pattern = item.split('=', 1)
            groups[label] = sorted(glob.glob(pattern))
        print(compare(groups, args.out or ROOT/'results'/'reports'/'compare.png', args.title))
    elif args.group:
        print(json.dumps(group(args.group, args.runs, args.out), indent=2))
    else:
        for run in args.runs:
            s = analyse(run)
            print(json.dumps({k: s[k] for k in ('run', 'validation')}, indent=2))


if __name__ == '__main__':
    main()
