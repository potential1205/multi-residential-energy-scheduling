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

# SCENARIO 1 (Summer TOU scenario) : epsilon = 0.01, days = 30, test_days = 30
# SCENARIO 2 (Winter TOU scenario) : epsilon = 0.01, days = 30, test_days = 30
# SCENARIO 3 (Cooling energy-dominant scenario) : epsilon = 0.1, days = 30, test_days = 365
# SCENARIO 4 (Heating energy-dominant scenario) : epsilon = 0.1, days = 30, test_days = 365
# Paper code budget was iteration = 5000 (scenarios 1-2) / 10000 (3-4); with the tuned schedule 200 suffices.

# parameters
SCENARIO = 4
epsilon = 0.1
iteration = 200          # passes over the 30 training days (6000 days); the paper code used 10000
days = 30                # length of the training window (days replayed in a loop)
train_start = 0          # first day of the training window (0 = January 1, as in the paper)
test_days = 365

# fix
buffer_limit = 20000
start_size = 5000
learning_rate = 0.0001
gamma = 0.98
batch_size = 32
Tf= 2
feature = 52 + 2*Tf
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

# Profiles in 0.1 kWh units. Legacy files ("x.npy") are the original integer data; processed sites
# ("collection/site", built by experiments/datasets.py) are stored in kWh and converted here.
def load_profile(name, kind):
    if name.endswith('.npy'):
        return np.load(f"data/{kind}/{name}")
    return np.load(f"data/processed/{name}/{kind}.npy").astype(np.float64) * 10

# train data load
print("\nSCENARIO : {}".format(SCENARIO))
if SCENARIO == 1 or SCENARIO == 2:
    test_load_data = load_data = np.tile(np.array([4]*24), (days, 1))
    test_generation_data = generation_data = np.tile(np.array([1]*24), (days, 1))
else:
    load_file_list = ["table1_load.npy"] + sorted(name for name in os.listdir("data/load") if name != "table1_load.npy")
    generation_file_list = ["generation.npy"]
    if SCENARIO == 3: n = 0
    elif SCENARIO ==  4: n = 2
    print(load_file_list[n])
    load_data = load_profile(load_file_list[n], 'load')[train_start:train_start+days]
    test_load_data = load_profile(load_file_list[n], 'load')[0:test_days]
    generation_data = load_profile(generation_file_list[0], 'generation')[train_start:train_start+days]
    test_generation_data = load_profile(generation_file_list[0], 'generation')[0:test_days]

print("")
print(len(load_data),len(load_data[0]))
print(len(generation_data),len(generation_data[0]))
print(len(test_load_data),len(test_load_data[0]))
print(len(test_generation_data),len(test_generation_data[0]))

T = np.eye(24)

winter_TOU = [5,5,5, 5,5,5, 5,15,15, 15,25,10, 10,10,10, 10,10,15, 15,5,5, 5,5,5]  # 겨울
summer_TOU = [5,5,5, 5,5,5, 5,10,10 ,10,10,15, 15,15,15, 15,15,10, 10,5,5,5,5,5]  #여름

cost_history = []
result_cost_history, result_battery_history, result_action_history = [],[],[]

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


def train():
    s, a, r, s_prime = memory.sample(batch_size)
    q_a = q(s).gather(1, a)
    with torch.no_grad():
        min_q_prime = q_target(s_prime)
    # Only grid purchases that keep s' within battery capacity and charging rate are admissible.
    battery = torch.clamp(s_prime[:, 0] - s_prime[:, 50] + s_prime[:, 51+Tf], max=battery_max).unsqueeze(1)
    rate = battery - s_prime[:, :1]
    grid = torch.arange(21)
    admissible = (battery + grid >= 0) & (battery + grid <= battery_max) & ((rate + grid).abs() <= battery_rate)
    Q_target = min_q_prime.masked_fill(~admissible, float('inf')).min(dim=1, keepdim=True).values
    target = r + gamma * Q_target
    loss = F.smooth_l1_loss(q_a, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def tariff(day):
    if SCENARIO == 1:
        return summer_TOU
    if SCENARIO == 2:
        return winter_TOU
    return summer_TOU if 90 <= day < 273 else winter_TOU  # summer tariff from April 1 to September 30

# state layout: [battery, demand charge, one-hot hour (24), TOU prices from now (24), load t..t+Tf, generation t..t+Tf]
def initialize_state(day=0, loads=None, gens=None):
    loads = load_data if loads is None else loads
    gens = generation_data if gens is None else gens
    state = np.zeros(feature)
    state[2:26] = T[0]
    state[26:50] = tariff(day)
    state[50:51+Tf] = loads[0][:Tf+1]
    state[51+Tf:52+2*Tf] = gens[0][:Tf+1]
    return state

def next_state(n_epi,time,battery,charge,day_charge,TOU,load_data,generation_data,cycle):
    state_prime = np.zeros(feature)
    state_prime[0] = battery + charge
    state_prime[1] = 0 if time == 23 else max(day_charge)
    state_prime[2:26] = T[(time + 1) % 24]
    state_prime[26:50] = np.roll(TOU, -(time + 1))
    hours = np.arange(time + 1, time + 2 + Tf)
    day_index = (n_epi + hours // 24) % cycle
    state_prime[50:51+Tf] = load_data[day_index, hours % 24]
    state_prime[51+Tf:52+2*Tf] = generation_data[day_index, hours % 24]
    return state_prime

def exploration(n_epi):
    progress = min(n_epi / epsilon_decay_days, 1.0)
    return epsilon + (epsilon_final - epsilon) * progress

def update(n_epi):
    if n_epi % (epochs/10) == 0 and n_epi != 0:
        print(sum(cost_history[n_epi-days:n_epi]))

    if memory.size() > start_size:
        for _ in range(updates_per_day):
            train()


#initialize
q = Qnet()
q_target = Qnet()
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()
optimizer = optim.Adam(q.parameters(), lr=learning_rate)

# train
state = initialize_state(train_start)
TOU = tariff(train_start)  # the training window is replayed with the tariff of its first day
for n_epi in tqdm(range(epochs)):
    costs, actions = 0, []
    for time in range(24):
        battery = min(state[0] - state[50] + state[51+Tf],battery_max)
        action = q.sample_action(torch.from_numpy(state).float(), exploration(n_epi), battery,battery-state[0])
        actions.append(action)
        cost = action * TOU[time] + (pD * max(actions) if time == 23 else 0)
        costs += cost
        state_prime = next_state(n_epi,time,battery,action,actions,TOU,load_data,generation_data,days)
        memory.put((state, action, cost / cost_scale, state_prime))
        state = state_prime
        if (n_epi * 24 + time) % freq == 0:
            q_target.load_state_dict(q.state_dict())

    cost_history.append(costs)
    update(n_epi)


# train time average cost
completed_blocks = len(cost_history) // days
block_costs = np.asarray(cost_history[:completed_blocks * days]).reshape(completed_blocks, days).sum(axis=1)
check_dqn = np.cumsum(block_costs) / np.arange(1, completed_blocks + 1)

plt.figure(figsize=(15,10))
plt.plot(range(completed_blocks), check_dqn, label='DQL time avgerage cost', color='r')
plt.xlabel('Epochs',fontsize=22)
plt.ylabel('Cost(won)',fontsize=22)
plt.legend()
plt.show()


# test
state = initialize_state(0, test_load_data, test_generation_data)
state[0] = 15
for n_epi in tqdm(range(test_days)):
    TOU = tariff(n_epi)
    costs, actions, batterys = 0, [], []
    for time in range(24):
        battery = min(state[0] - state[50] + state[51+Tf], battery_max)
        action = q.sample_action(torch.from_numpy(state).float(), 0, battery,battery-state[0])
        actions.append(action)
        batterys.append(state[0])
        cost = action * TOU[time] + (pD * max(actions) if time == 23 else 0)
        costs += cost
        state_prime = next_state(n_epi,time,battery,action,actions,TOU,test_load_data,test_generation_data,test_days)
        state = state_prime
    result_cost_history.append(costs)
    result_action_history.append(actions)
    result_battery_history.append(batterys)

print(sum(result_cost_history))
print(result_cost_history)
print(result_action_history)

test_actions = np.asarray(result_action_history)
TOU_cost = int((np.array([tariff(day) for day in range(test_days)]) * test_actions).sum())
DC_cost = int(test_actions.max(axis=1).sum() * pD)

print("Total_Cost : {}, 평균 : {}".format(TOU_cost+DC_cost,(TOU_cost+DC_cost)//(test_days//30)))
print("TOU_Cost : {}, 평균 : {}".format(TOU_cost,TOU_cost//(test_days//30)))
print("DC_Cost : {}, 평균 : {}".format(DC_cost,DC_cost//(test_days//30)))

# scheduling result (first test day)
x = range(24)
TOU = tariff(0)
plt.plot(x, TOU, label='TOU', color='gray')
plt.fill_between(x, TOU, color='lightgray')
plt.plot(x, result_battery_history[0], linewidth=3, label='HM_charge', color='Red')
plt.plot(x, result_action_history[0], linewidth=3, label='DQL', color='Orange')
plt.plot(x, generation_data[0], '--', label='Generation', color='gray')
plt.plot(x, load_data[0], '-', label='Load', color='black')
plt.xticks(np.arange(0, 24))
plt.yticks(np.arange(0, 41, 2))
plt.grid(True)
plt.show()
