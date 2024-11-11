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

- Root Directory: Contains core algorithm files for energy scheduling and federated learning models.
- `data/`: Contains datasets and configurations used for training and validation.
    - `load/`: Power demand data.
    - `generation/`: Renewable energy generation data.

## Getting Started

### Prerequisites

- Python: 3.9.6
- Required libraries: `numpy`, `pandas`, `torch` (for deep learning models)

### Installation

Clone the repository:

```bash
git clone https://github.com/username/multi-residential-energy-scheduling.git
cd multi-residential-energy-scheduling
```
