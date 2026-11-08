# $ pip install -r requirements.txt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import matplotlib.pyplot as plt
import random
import math
from tqdm import tqdm
import os
from datetime import datetime
import copy

# TOUs, DCs setting
# Fig 5 : TOU=[s,w,s,w] DC=[20,25,30,35] (DRL4 vs FRL4)
# Fig 6 : TOU=[s,w,s,w,s,w] DC=[20,25,30,35,20,25] (4+2)
# Fig 7 : 1,2,8 TOUs=[s,s,s,s,w,w,w,w] DCs=[20,25,30,35,20,25,30,35] / 4 TOUs=[s,s,w,w,s,s,w,w] DCs=[30,35,20,25,20,25,30,35] (1,2,4,8)

# parameters
N_new = N = 4
NEW = 0
CO = 1

epsilon = 0.1
iteration = 200          # passes over the 30 training days (6000 days); the paper code used 10000
learning_rate = 0.00001
days = 30
start_size = 5000
gamma = 0.98
buffer_limit = 20000
batch_size = 32
Tf= 2
feature = 52 + 2*Tf +1
battery_max = 40
freq = 240               # target network sync period in hours (paper: 20)
epochs = days * iteration
pD=30
battery_rate=20
# learning schedule. Paper settings: updates_per_day=1, cost_scale=1, freq=20, iteration=10000.
# The tuned values reach the paper's 300k-day result in ~2-3k days (see README, "Hyperparameter study").
updates_per_day = 24      # gradient steps after each simulated day
cost_scale = 100          # costs are divided by this before entering the replay buffer
epsilon_final = epsilon   # epsilon decays linearly from `epsilon` to this value ...
epsilon_decay_days = 1    # ... over this many training days

T = np.eye(24)
w= [5,5,5, 5,5,5, 5,15,15, 15,25,10, 10,10,10, 10,10,15, 15,5,5, 5,5,5]  # 겨울
s = [5,5,5, 5,5,5, 5,10,10 ,10,10,15, 15,15,15, 15,15,10, 10,5,5,5,5,5]  #여름

TOUs = [s,w,s,w,s,w,s,w]
DCs = [20,25,30,35,20,25,30,35]

# train data load
load_datas, generation_datas = [], []

load_file_list = ["table1_load.npy"] + sorted(name for name in os.listdir("data/load") if name != "table1_load.npy")
generation_file_list = ["generation.npy"]

for n in range(N+NEW):
    print(load_file_list[n])
    load_data = (np.load(f"data/load/{load_file_list[n]}")[0:days])
    generation_data = (np.load(f"data/generation/{generation_file_list[0]}")[0:days])
    load_datas.append(load_data)
    generation_datas.append(generation_data)

print(len(load_datas),len(load_datas[0]),len(load_datas[0][0]))
print(len(generation_datas),len(generation_datas[0]),len(generation_datas[0][0]))

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        s, a, r, s_prime = zip(*random.sample(self.buffer, n))
        return (torch.from_numpy(np.asarray(s, dtype=np.float32)),
                torch.from_numpy(np.asarray(a, dtype=np.int64)).unsqueeze(1),
                torch.from_numpy(np.asarray(r, dtype=np.float32)).unsqueeze(1),
                torch.from_numpy(np.asarray(s_prime, dtype=np.float32)))

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(feature, 512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 21)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    @torch.no_grad()
    def sample_action(self, state, epsilon, battery, rate):
        # The admissible integer actions form one contiguous interval.
        lower = max(0, math.ceil(-battery), math.ceil(-battery_rate - rate))
        upper = min(20, math.floor(battery_max - battery), math.floor(battery_rate - rate))
        if lower > upper:
            raise ValueError(f"No feasible action: battery={battery}, rate={rate}")
        if random.random() < epsilon:
            return random.randint(lower, upper)
        out = self.forward(state)
        return lower + int(torch.argmin(out[lower:upper + 1]).item())


def train(n):
    s, a, r, s_prime = memory[n].sample(batch_size)
    q_a = q[n](s).gather(1, a)
    with torch.no_grad():
        min_q_prime = q_target[n](s_prime)
    # Only grid purchases that keep s' within battery capacity and charging rate are admissible.
    battery = torch.clamp(s_prime[:, 0] - s_prime[:, 50] + s_prime[:, 51+Tf], max=battery_max).unsqueeze(1)
    rate = battery - s_prime[:, :1]
    grid = torch.arange(21)
    admissible = (battery + grid >= 0) & (battery + grid <= battery_max) & ((rate + grid).abs() <= battery_rate)
    Q_target = min_q_prime.masked_fill(~admissible, float('inf')).min(dim=1, keepdim=True).values
    target = r + gamma * Q_target
    loss = F.smooth_l1_loss(q_a, target)
    optimizer = optimizers[n]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# state layout: [battery, demand charge, one-hot hour (24), TOU prices from now (24),
#                load t..t+Tf, generation t..t+Tf, client demand-charge price]
def initialize_state(n):
    state = np.zeros(feature)
    state[2:26] = T[0]
    state[26:50] = TOUs[n]
    state[50:51+Tf] = load_datas[n][0][:Tf+1]
    state[51+Tf:52+2*Tf] = generation_datas[n][0][:Tf+1]
    state[-1] = DCs[n]
    return state

def next_state(n,n_epi,time,battery,charge,day_charge):
    state_prime = np.zeros(feature)
    state_prime[0] = battery + charge
    state_prime[1] = 0 if time == 23 else max(day_charge)
    state_prime[2:26] = T[(time + 1) % 24]
    state_prime[26:50] = np.roll(TOUs[n], -(time + 1))
    hours = np.arange(time + 1, time + 2 + Tf)
    day_index = (n_epi + hours // 24) % days
    state_prime[50:51+Tf] = load_datas[n][day_index, hours % 24]
    state_prime[51+Tf:52+2*Tf] = generation_datas[n][day_index, hours % 24]
    state_prime[-1] = DCs[n]
    return state_prime

def get_grad(before_q,after_q):
    return {key: after_q[key] - before_q[key] for key in after_q}

def agg_grad(global_q,grads,weights=None):
    if not grads:
        raise ValueError("Cannot aggregate an empty client set")
    weights = np.ones(len(grads)) if weights is None else np.asarray(weights, dtype=float)
    if len(weights) != len(grads) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid aggregation weights")
    weights = weights / weights.sum()
    w_avg = copy.deepcopy(global_q)
    for key in w_avg.keys():
        for i in range(len(grads)):
            w_avg[key] += float(weights[i]) * grads[i][key]
    return w_avg

def federated_learning():
    grads = [get_grad(q_before[i].state_dict(), q[i].state_dict()) for i in range(N_new)]
    agg_model = agg_grad(q_global.state_dict(), grads, experience_counts[:N_new])
    q_global.load_state_dict(agg_model)
    for i in range(N_new):
        q[i].load_state_dict(agg_model)
        q_target[i].load_state_dict(agg_model)

def exploration(n_epi):
    progress = min(n_epi / epsilon_decay_days, 1.0)
    return epsilon + (epsilon_final - epsilon) * progress

def update(n_epi,n):
    if n_epi % (epochs/10) == 0:
        print(sum(cost_history[n][-days:]))

    if memory[n].size() > start_size:
        for _ in range(updates_per_day):
            train(n)

cost_history = [[] for _ in range(N+NEW)]
action_history = [[] for _ in range(N+NEW)]
experience_counts = [0] * (N+NEW)
q, q_target, memory, q_before, optimizers, q_global = [], [], [], [], [], Qnet()

# Every client starts from the global model. Construction order is kept so the RNG stream is unchanged.
for n in range(N+NEW):
    q.append(Qnet())
    optimizers.append(optim.Adam(q[n].parameters(), lr=learning_rate))
    q_target.append(Qnet())
    memory.append(ReplayBuffer())
    q[n].load_state_dict(q_global.state_dict())
    q_target[n].load_state_dict(q_global.state_dict())
    q_before.append(Qnet())


# train
state = [initialize_state(n) for n in range(N+NEW)]
for n_epi in tqdm(range(epochs)):
    if epochs//2 == n_epi and NEW != 0:
        print("start new EMS")
        N_new += NEW
        for r in range(NEW):
            q[N+r].load_state_dict(q_global.state_dict())
            q_target[N+r].load_state_dict(q_global.state_dict())

    for n in range(N_new):
        costs, actions = 0, []
        q_before[n].load_state_dict(q[n].state_dict())
        for time in range(24):
            battery = min(state[n][0] - state[n][50] + state[n][51+Tf],battery_max)
            action = q[n].sample_action(torch.from_numpy(state[n]).float(), exploration(n_epi), battery,battery-state[n][0])
            actions.append(action)
            cost = action * TOUs[n][time] + (DCs[n] * max(actions) if time == 23 else 0)
            costs += cost
            state_prime = next_state(n,n_epi,time,battery,action,actions)
            memory[n].put((state[n], action, cost / cost_scale, state_prime))
            experience_counts[n] += 1
            state[n] = state_prime
            if (n_epi * 24 + time) % freq == 0:
                q_target[n].load_state_dict(q[n].state_dict())

        cost_history[n].append(costs)
        action_history[n].append(actions)
        update(n_epi,n)

    if CO == 1:
        federated_learning()


# train time average cost
for n in range(N+NEW):
    completed_blocks = len(cost_history[n]) // days
    usable = np.asarray(cost_history[n][:completed_blocks * days])
    block_costs = usable.reshape(completed_blocks, days).sum(axis=1)
    check_dqn = np.cumsum(block_costs) / np.arange(1, completed_blocks + 1)
    plt.figure(figsize=(15,10))
    plt.plot(range(completed_blocks), check_dqn, label='DQL time avgerage cost', color='r')
    plt.xlabel('Epochs',fontsize=22)
    plt.ylabel('Cost(won)',fontsize=22)
    plt.legend()
    plt.show()


def save_data():
    directory = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
    path = 'log/' + directory
    os.makedirs(path + '/models', exist_ok=True)

    info = dict(N=N, Tf=Tf, gamma=gamma, T_target=freq, T_FRL=1, learning_rate=learning_rate,
                batch_size=batch_size, start_size=start_size, buffer_limit=buffer_limit, episodes=iteration,
                number_of_action=21, epsilon=epsilon, NEW=NEW, feature_size=feature, train_days=days,
                battery_max=battery_max, battery_rate=battery_rate, summer_TOU=s, winter_TOU=w,
                TOUs=TOUs, DCs=DCs, hidden_layer=3, number_of_node=512, activation_function="relu")
    with open(path + '/_parameters.txt', 'w', encoding='UTF-8') as f:
        for code, name in info.items():
            f.write(f'{code} : {name}\n')

    np.save(path + '/train_cost_historys.txt', cost_history[0:N])
    np.save(path + '/train_action_historys.txt', action_history[0:N])
    if NEW != 0:
        np.save(path + '/new_train_cost_history.txt', cost_history[N:])
        np.save(path + '/new_train_action_history.txt', action_history[N:])

    for n in range(N+NEW):
        torch.save(q[n].state_dict(), path + f'/models/local{n+1}.pt')
    if CO == 1:
        torch.save(q_global.state_dict(), path + '/models/global.pt')

save_data()
