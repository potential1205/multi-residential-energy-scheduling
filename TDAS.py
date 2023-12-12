# $ pip install -r requirements.txt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os

# SCENARIO 1 (Summer TOU scenario) : iteration = 5000, epsilon = 0.01, days = 30, test_days = 30
# SCENARIO 2 (Winter TOU scenario) : iteration = 5000, epsilon = 0.01, days = 30, test_days = 30
# SCENARIO 3 (Cooling energy-dominant scenario) : iteration = 10000, epsilon = 0.1, days = 30, test_days = 360
# SCENARIO 4 (Heating energy-dominant scenario) : iteration = 10000, epsilon = 0.1, days = 30, test_days = 360

# parameters
SCENARIO = 4
epsilon = 0.1
iteration = 10000
days = 30
test_days = 360

# fix
buffer_limit = 20000
start_size = 5000
learning_rate = 0.0001
gamma = 0.98
batch_size = 32
Tf= 2
feature = 52 + 2*Tf
battery_max = 40
freq = 20
epochs = days * iteration
pD=30
battery_rate=20

# train data load
print("\nSCENARIO : {}".format(SCENARIO))
if SCENARIO == 1 or SCENARIO == 2:
    test_load_data = load_data = np.tile(np.array([4]*24), (days, 1))
    test_generation_data = generation_data = np.tile(np.array([1]*24), (days, 1))
else:
    load_file_list = os.listdir("data/load")
    generation_file_list = os.listdir("data/generation")
    if SCENARIO == 3: n = 0
    elif SCENARIO ==  4: n = 2
    print(load_file_list[n])
    load_data = (np.load(f"data/load/{load_file_list[n]}")[0:days])
    test_load_data = (np.load(f"data/load/{load_file_list[n]}")[0:test_days])
    generation_data = (np.load(f"data/generation/{generation_file_list[0]}")[0:days])
    test_generation_data = (np.load(f"data/generation/{generation_file_list[0]}")[0:test_days])

print("")
print(len(load_data),len(load_data[0]))
print(len(generation_data),len(generation_data[0]))
print(len(test_load_data),len(test_load_data[0]))
print(len(test_generation_data),len(test_generation_data[0]))

T = np.array(pd.get_dummies(np.array([0,1,2,3,4,5,6,7,8,9,10,11,
                                      12,13,14,15,16,17,18,19,20,21,22,23])))

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
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
        return torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(np.array(a_lst),dtype=torch.int64), torch.tensor(np.array(r_lst),dtype=torch.int64),torch.tensor(np.array(s_prime_lst),dtype=torch.float)

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

    def sample_action(self, state, epsilon, battery, rate):
        out = self.forward(state)
        q_list = [out[action].item() if (0 <= battery + action <= battery_max) and (
                    -battery_rate <= rate + action <= battery_rate) else np.Inf for action in range(0, 21, 1)]
        if random.random() < epsilon:
            return random.choice([i for i in range(len(q_list)) if q_list[i] != np.Inf])
        else:
            return q_list.index(min(q_list))

# def train():
#     s, a, r, s_prime = memory.sample(batch_size)
#     q_a = q(s).gather(1,a)
#     min_q_prime = q_target(s_prime)
#     Q_target = list()
#     for i in range(batch_size):
#         battery = min(s_prime[i][0].item() - s_prime[i][50].item() + s_prime[i][51+Tf].item(), battery_max)
#         rate = battery-s_prime[i][0].item()
#         q_list = []
#         for j in range(0, 21, 1):
#             if (0<= battery + j <= battery_max) and (-battery_rate <= rate + j <= battery_rate):
#                 q_list.append(min_q_prime[i][j].item())
#             else:
#                 q_list.append(np.Inf)
#         Q_target.append(min(q_list))
#
#     Q_target = torch.tensor((Q_target)).resize(batch_size,1)
#     target = r + gamma * Q_target
#     loss = F.smooth_l1_loss(q_a, target)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
def train():
    s, a, r, s_prime = memory.sample(batch_size)
    q_a = q(s).gather(1, a)
    min_q_prime = q_target(s_prime)
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
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def initialize_state():
    TOU = summer_TOU if SCENARIO == 1 else winter_TOU
    state = np.zeros(feature)
    state[0] = 0    # battery level
    state[1] = 0    # demand charge
    for i in range(24):
        state[2 + i] = T[0][i] # time (one-hot encoding)
        state[26 + i] = TOU[i] # TOU price
    for i in range(Tf+1):
        state[50+i] = load_data[0][i] # load (t~t+Tf)
        state[51+Tf+i] = generation_data[0][i] # generation (t~t+Tf)
    return state

def next_state(n_epi,time,battery,charge,day_charge,TOU,load_data,generation_data,cycle):
    state_prime = np.zeros(feature)
    state_prime[0] = battery + charge
    if time==23: state_prime[1] = 0
    else: state_prime[1] = max(day_charge)
    for i in range(24):
        state_prime[2 + i] = T[(time + 1) % 24][i]
        state_prime[26 + i] = TOU[(time+1+i)%24]
    for i in range(Tf+1):
        if time+1+i >= 24:
            state_prime[50+i] = load_data[(n_epi+1)%cycle][(time+1+i)%24]
            state_prime[51+Tf+i] = generation_data[(n_epi+1)%cycle][(time+1+i)%24]
        else:
            state_prime[50+i] = load_data[(n_epi)%cycle][(time+1+i)%24]
            state_prime[51+Tf+i] = generation_data[(n_epi)%cycle][(time+1+i)%24]

    return state_prime

def update(n_epi):
    if n_epi % (epochs/10) == 0 and n_epi != 0:
        print(sum(cost_history[n_epi-days:n_epi]))

    if memory.size() > start_size:
        train()

    if n_epi % freq == 0 and n_epi != 0:
        q_target.load_state_dict(q.state_dict())

#initialize
q = Qnet()
q_target = Qnet()
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()
optimizer = optim.Adam(q.parameters(), lr=learning_rate)

# train
state = initialize_state()
TOU = summer_TOU if SCENARIO == 1 else winter_TOU
for n_epi in tqdm(range(epochs)):
    costs, actions = 0, []
    for time in range(24):
        battery = min(state[0] - state[50] + state[51+Tf],battery_max)
        action = q.sample_action(torch.from_numpy(state).float(), epsilon, battery,battery-state[0])
        actions.append(action)
        cost = action * TOU[time] + (pD * max(actions) if time == 23 else 0)
        costs += cost
        state_prime = next_state(n_epi,time,battery,action,actions,TOU,load_data,generation_data,days)
        memory.put((state, action, cost, state_prime))
        state = state_prime

    cost_history.append(costs)
    update(n_epi)


# train time average cost
check_dqn, cum_cost = np.array([]), 0

for i in range(iteration):
  cum_cost += sum(cost_history[i*days : i*days+days])
  check_dqn = np.append(check_dqn,cum_cost/(i+1))

plt.figure(figsize=(15,10))
x = range(0, iteration)
y1 = [check_dqn[v] for v in x]
plt.plot(x,y1,label='DQL time avgerage cost', color='r')
plt.xlabel('Epochs',fontsize=22)
plt.ylabel('Cost(won)',fontsize=22)
plt.legend()
plt.show()


# test
state = initialize_state()
state[0] = 15
for n_epi in tqdm(range(test_days)):
    if SCENARIO == 1:
        TOU = summer_TOU
    elif SCENARIO == 2:
        TOU = winter_TOU
    elif 90 <= n_epi < 270:
        TOU = summer_TOU
    else:
        TOU = winter_TOU

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

TOU_cost = 0
DC_cost = 0
temp_action = []
for day in range(test_days):
    if SCENARIO == 1:
        TOU = summer_TOU
    elif SCENARIO == 2:
        TOU = winter_TOU
    elif 90 <= day < 270:
        TOU = summer_TOU
    else:
        TOU = winter_TOU

    day_action = []
    for t in range(24):
        action = result_action_history[day][t]
        day_action.append(action)
        TOU_cost += TOU[t] * action
    DC_cost += max(day_action) * pD
    temp_action.append(day_action)

print("Total_Cost : {}, 평균 : {}".format(TOU_cost+DC_cost,(TOU_cost+DC_cost)//(test_days//30)))
print("TOU_Cost : {}, 평균 : {}".format(TOU_cost,TOU_cost//(test_days//30)))
print("DC_Cost : {}, 평균 : {}".format(DC_cost,DC_cost//(test_days//30)))

# scheduling result
for i in range(1):
    if SCENARIO == 1:
        TOU = summer_TOU
    elif SCENARIO == 2:
        TOU = winter_TOU
    elif 90 <= n_epi < 270:
        TOU = summer_TOU
    else:
        TOU = winter_TOU

    x = range(0, 24)
    y1 = [v for v in load_data[i]]
    y2 = [v for v in result_action_history[i]]
    y3 = [v for v in TOU[0:24]]
    y4 = [v for v in result_battery_history[i]]
    y5 = [v for v in generation_data[i]]

    plt.plot(x, y3, label='TOU', color='gray')
    plt.fill_between(x[0:24], y3[0:24], color='lightgray')
    plt.plot(x, y4, linewidth=3, label='HM_charge', color='Red')
    plt.plot(x, y2, linewidth=3 ,label='DQL', color='Orange')
    plt.plot(x, y5, '--',label='Generation',color='gray')
    plt.plot(x, y1,'-', label='Load', color='black')
    plt.xticks(np.arange(0, 24))
    plt.yticks([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40])
    plt.grid(True)
    plt.show()