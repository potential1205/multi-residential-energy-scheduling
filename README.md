# Multi-Residential Energy Scheduling Under Time-of-Use and Demand Charge Tariffs With Federated Reinforcement Learning

## Introduction

This repository accompanies the paper **"[Multi-Residential Energy Scheduling Under Time-of-Use and Demand Charge Tariffs With Federated Reinforcement Learning](https://ieeexplore.ieee.org/document/10057441)"**, published in IEEE Transactions on Smart Grid, a leading journal in the electrical and electronics engineering field. This study focuses on reducing energy costs for multiple residences by utilizing a novel Federated Reinforcement Learning (FRL) approach that effectively schedules energy across units with diverse energy demands and resources, considering both time-of-use (TOU) and demand charge (DC) tariffs.

## Abstract

The research introduces a TOU and DC-aware energy scheduling (TDAS) algorithm based on deep reinforcement learning (DRL). The algorithm manages the on-grid energy consumption of individual energy management systems (EMSs) without requiring prior information on uncertainties. For multiple EMSs, a cooperative version of the algorithm, Co-TDAS, is implemented using Federated Reinforcement Learning, allowing EMSs to collaboratively optimize energy costs in a privacy-preserving manner.

## Key Contributions

- **TOU and DC Tariff Optimization**: Develops a TDAS algorithm to manage energy scheduling for both TOU and DC tariffs, providing significant cost savings.
- **Federated Learning Integration**: Applies federated reinforcement learning to enable cooperative learning among multiple EMSs while preserving data privacy.
- **EMS-Agnostic Policy Design**: Introduces a universal energy scheduling policy applicable across various EMS configurations and environments.

## Research Methodology

The study addresses energy scheduling through:

1. Developing a DRL-based TDAS policy for single EMS energy optimization.
2. Extending to a federated reinforcement learning-based Co-TDAS algorithm for multiple EMS cooperation.
3. Testing and comparing against state-of-the-art models, such as MPC and TAS, using real datasets for validation.

## Experimental Results

Simulation results demonstrate:

- **Cost Efficiency**: The TDAS algorithm achieves cost performance on par with or better than existing models, even under uncertain conditions.
- **Scalability and Adaptability**: The Co-TDAS model quickly adapts to diverse EMS conditions and accelerates learning through cooperative federated learning.

## Repository Structure

- `TDAS.py`, `Co-TDAS.py`: single-EMS and federated (Co-TDAS) training scripts. They run standalone and use the tuned defaults below.
- `MPC.py`: the MPC baseline (needs `requirements-mpc.txt`). `python -m experiments.run_mpc` runs it headless on any dataset; see `docs/mpc-comparison.md`.
- `experiments/`: seeded runner, live telemetry, reports, suites and tests.
  - `run_original.py`: runs a training script headless with a seed, a preset and per-run artifacts.
  - `suite.py` + `plans/*.json`: parallel experiment plans; every study in `docs/` is one plan file.
  - `report.py`: per-run figures and metrics, seed-group summaries and configuration comparisons.
  - `dashboard_server.py` + `dashboard/`: local live dashboard.
  - `status.py`: run lifecycle (queued, running, stalled, completed, stopped, failed, interrupted).
- `data/`: `load/` demand profiles and `generation/` PV generation (365 x 24, 0.1 kWh units).
- `docs/hyperparameter-study.md`: convergence measurements, tuning results and the paper cross-check.

## Getting Started

Python 3.12 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Train

```bash
# Tuned defaults: 6000 training days, about 10 minutes on one CPU thread
.venv/bin/python -m experiments.run_original --scenario 3 --seed 0 --eval-every 250
# Exact paper settings (bit-identical to the original code for the same seed)
.venv/bin/python -m experiments.run_original --preset paper --scenario 3 --seed 0 --eval-every 5000
# Federated Co-TDAS with 4 EMSs
.venv/bin/python -m experiments.run_original --algorithm Co-TDAS --clients 4 --seed 0
```

Each run writes `results/runs/<run>/` with the config, the executed source, metrics (`metrics/events.jsonl`), checkpoints (`best-validation-*.pt` is selected on days 30-59, never on test days), status and a SHA-256 manifest. `Ctrl-C` or `SIGTERM` stops a run cleanly and saves partial results as `stopped-*`.

| preset | updates per day | target sync | cost scale | width | training days |
|---|---|---|---|---|---|
| `tuned` (default) | 24 | 240 h | 1/100 | 512 | 6000 |
| `fast` | 12 | 240 h | 1/100 | 256 | 6000 |
| `paper` | 1 | 20 h | 1 | 512 | 150k (toy) / 300k |

`tuned` reaches the paper settings' 300k-day result in about 2-3k days. See [docs/hyperparameter-study.md](docs/hyperparameter-study.md).

### Monitor and analyse

```bash
# Live dashboard at http://127.0.0.1:8765
.venv/bin/python -m experiments.dashboard_server
# Figures and metrics for one run, a seed group, or several configurations
.venv/bin/python -m experiments.report results/runs/<run>
.venv/bin/python -m experiments.report --group my-config results/runs/<prefix>-seed*
.venv/bin/python -m experiments.report --compare --out compare.png "paper=results/runs/a-*" "tuned=results/runs/b-*"
# Run a whole plan in parallel
.venv/bin/python -m experiments.suite --plan experiments/plans/tuning-stage3-v1.json --workers 6
```

### Tests

```bash
.venv/bin/python -m unittest experiments.test_core experiments.test_runtime
node experiments/test_dashboard.mjs
```
