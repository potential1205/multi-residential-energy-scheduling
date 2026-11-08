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
iteration = 10000
learning_rate = 0.00001
days = 30
start_size = 5000
gamma = 0.98
buffer_limit = 20000
batch_size = 32
Tf= 2
feature = 52 + 2*Tf +1
battery_max = 40
freq = 20
epochs = days * iteration
pD=30
battery_rate=20

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
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
        return (torch.from_numpy(np.asarray(s_lst, dtype=np.float32)), torch.from_numpy(np.asarray(a_lst, dtype=np.int64)), torch.from_numpy(np.asarray(r_lst, dtype=np.float32)), torch.from_numpy(np.asarray(s_prime_lst, dtype=np.float32)))

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
    battery = torch.clamp(s_prime[:, 0] - s_prime[:, 50] + s_prime[:, 51+Tf], max=battery_max)
    rate = battery - s_prime[:, 0]
    q_list = torch.where((0 <= battery.unsqueeze(-1) + torch.arange(21)) &
                         ((battery.unsqueeze(-1) + torch.arange(21)) <= battery_max) &
                         ((-battery_rate <= rate.unsqueeze(-1) + torch.arange(21)) &
                          (rate.unsqueeze(-1) + torch.arange(21) <= battery_rate)),
                         min_q_prime, torch.tensor(float('inf')))

    Q_target, _ = torch.min(q_list, dim=1, keepdim=True)
    target = r + gamma * Q_target
    loss = F.smooth_l1_loss(q_a, target)
    optimizer = optimizers[n]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# def train(n):
#     s, a, r, s_prime = memory[n].sample(batch_size)
#     q_a = q[n](s).gather(1,a)
#     min_q_prime = q_target[n](s_prime)
#     Q_target = list()
#     for i in range(batch_size):
#         battery = min(s_prime[i][0].item() - s_prime[i][50].item() + s_prime[i][51+Tf].item(),battery_max)
#         rate = battery-s_prime[i][0].item()
#         q_list = []
#         for j in range(0, 21, 1):
#             if (0<= battery + j <= battery_max) and (-battery_rate <= rate + j <= battery_rate):
#                 q_list.append(min_q_prime[i][j].item())
#             else:
#                 q_list.append(np.inf)
#         Q_target.append(min(q_list))
#
#     optimizer = optim.Adam(q[n].parameters(), lr=learning_rate)
#     Q_target = torch.tensor((Q_target)).resize(batch_size,1)
#     target = r + gamma * Q_target
#     loss = F.smooth_l1_loss(q_a, target)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

def initialize_state(n):
    state = np.zeros(feature)
    state[0] = 0    # battery level
    state[1] = 0    # demand charge
    for i in range(24):
        state[2 + i] = T[0][i] # time (one-hot encoding)
        state[26 + i] = TOUs[n][i] # TOU price
    for i in range(Tf+1):
        state[50+i] = load_datas[n][0][i] # load (t~t+Tf)
        state[51+Tf+i] = generation_datas[n][0][i] # generation (t~t+Tf)
    state[51+Tf+i+1] = DCs[n]

    return state

def next_state(n,n_epi,time,battery,charge,day_charge,TOU,load_datas,generation_datas):
    state_prime = np.zeros(feature)
    state_prime[0] = battery + charge
    if time==23: state_prime[1] = 0
    else: state_prime[1] = max(day_charge)
    for i in range(24):
        state_prime[2 + i] = T[(time + 1) % 24][i]
        state_prime[26 + i] = TOUs[n][(time+1+i)%24]
    for i in range(Tf+1):
        if time+1+i >= 24:
            state_prime[50+i] = load_datas[n][(n_epi+1)%days][(time+1+i)%24]
            state_prime[51+Tf+i] = generation_datas[n][(n_epi+1)%days][(time+1+i)%24]
        else:
            state_prime[50+i] = load_datas[n][(n_epi)%days][(time+1+i)%24]
            state_prime[51+Tf+i] = generation_datas[n][(n_epi)%days][(time+1+i)%24]

    state_prime[51+Tf+i+1] = DCs[n]

    return state_prime

def get_grad(before_q,after_q):
    output = copy.deepcopy(after_q)
    for key in output.keys():
        output[key] = (after_q[key]-before_q[key])
    return output

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
    grads = []
    for i in range(N_new):
        grads.append(get_grad(q_before[i].state_dict(), q[i].state_dict()))
    agg_model = agg_grad(q_global.state_dict(), grads, experience_counts[:N_new])
    q_global.load_state_dict(agg_model)
    for i in range(N_new):
        q[i].load_state_dict(agg_model)
        q_target[i].load_state_dict(agg_model)

def update(n_epi,n):
    if n_epi % (epochs/10) == 0:
        leng = len(cost_history[n])
        print(sum(cost_history[n][leng-days:leng]))

    if memory[n].size() > start_size:
        train(n)

cost_history,action_history,battery_history=[],[],[]
q, q_target, memory, q_before, q_global = [], [], [], [], Qnet()
optimizers = []
experience_counts = [0 for _ in range(N+NEW)]

for n in range(N+NEW):
    q.append(Qnet())
    optimizers.append(optim.Adam(q[n].parameters(), lr=learning_rate))
    q_target.append(Qnet())
    memory.append(ReplayBuffer())
    q[n].load_state_dict(q_global.state_dict())
    q_target[n].load_state_dict(q_global.state_dict())
    q_before.append(Qnet())
    cost_history.append([])
    action_history.append([])


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
            action = q[n].sample_action(torch.from_numpy(state[n]).float(), epsilon, battery,battery-state[n][0])
            actions.append(action)
            cost = action * TOUs[n][time] + (DCs[n] * max(actions) if time == 23 else 0)
            costs += cost
            state_prime = next_state(n,n_epi,time,battery,action,actions,TOUs[n],load_datas,generation_datas)
            memory[n].put((state[n], action, cost, state_prime))
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
    x = range(completed_blocks)

    plt.figure(figsize=(15,10))
    y1 = [check_dqn[v] for v in x]
    plt.plot(x,y1,label='DQL time avgerage cost', color='r')
    plt.xlabel('Epochs',fontsize=22)
    plt.ylabel('Cost(won)',fontsize=22)
    plt.legend()
    plt.show()


def save_data():
    directory = str(datetime.now().strftime("%Y.%m.%d %H-%M-%S"))
    path = 'log/' + directory
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Failed to create the directory.")

    info = dict()
    info['N'] = N
    info['Tf'] = Tf
    info['gamma'] = gamma
    info['T_target'] = 20
    info['T_FRL'] = 1
    info['learning_rate'] = learning_rate
    info['batch_size'] = batch_size
    info['start_size'] = start_size
    info['buffer_limit'] = buffer_limit
    info['episodes'] = iteration
    info['number_of_action'] = 21
    info['epsilon'] = epsilon
    info['NEW'] = NEW
    info['feature_size'] = feature
    info['train_days'] = days
    info['battery_max'] = battery_max
    info['battery_rate'] = battery_rate
    info['summer_TOU'] = s
    info['winter_TOU'] = w
    info['TOUs'] = TOUs
    info['DCs'] = DCs

    info['hidden_layer'] = 3
    info['number_of_node'] = 512
    info['activation_function'] = "relu"

    with open(path + '/_parameters.txt', 'w', encoding='UTF-8') as f:
        for code, name in info.items():
            f.write(f'{code} : {name}\n')

    new_train_cost_history = []
    new_train_action_history = []

    for n in range(NEW):
        new_train_cost_history.append(cost_history[N+n])
        new_train_action_history.append(action_history[N+n])

    np.save(path + f'/train_cost_historys.txt', cost_history[0:N])
    np.save(path + f'/train_action_historys.txt', action_history[0:N])

    if NEW != 0:
        np.save(path + f'/new_train_cost_history.txt', cost_history[N:])
        np.save(path + f'/new_train_action_history.txt', action_history[N:])

    path = 'log/' + directory + '/models'
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Failed to create the models directory.")

    for n in range(N+NEW):
        torch.save(q[n].state_dict(),path + f'/local{n+1}.pt')
    if CO == 1:
        torch.save(q_global.state_dict(), path + f'/global.pt')

save_data()
