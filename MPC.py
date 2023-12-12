import do_mpc
from casadi import *
from tqdm import trange
import os
import numpy as np

# SCENARIO = 1 (Summer TOU scenario)
# SCENARIO = 2 (Winter TOU scenario)
# MPC-24 noise_max(0),n_horizon(30)
# MPC-6 noise_max(0), n_horizon(6)
# MPC-24n noise_max(0.6), n_horizon(24), linspace_noise_max(2)

# SCENARIO = 3 (Cooling energy-dominant scenario)
# SCENARIO = 4 (Heating energy-dominant scenario)
# MPC-24 : noise_max(0),n_horizon(24)
# MPC-6 : noise_max(0), n_horizon(6)
# MPC-24n : noise_max(0.6), n_horizon(24), linspace_noise_max(4)

# parameters
SCENARIO = 1
pD = 30
n_horizon = 30 # 30 or 6 or 24

scaling_factor = 100  # to avoid numerical issues
noise_max = 0 * scaling_factor #
p_d = 0.3 * scaling_factor
B_init = 1.5 * scaling_factor
B_max = 4 * scaling_factor
B_rate = 2 * scaling_factor
C_max = 2 * scaling_factor
days_padding = 2  # to avoid MPC implementation issues


# data load
load_file_list = os.listdir("data/load")
generation_file_list = os.listdir("data/generation")
demand_energy_data, renewable_energy_data = [],[]
summer_TOU = [5,5,5, 5,5,5, 5,10,10 ,10,10,15, 15,15,15, 15,15,10, 10,5,5,5,5,5]
winter_TOU = [5,5,5, 5,5,5, 5,15,15, 15,25,10, 10,10,10, 10,10,15, 15,5,5, 5,5,5]

if SCENARIO == 1 or SCENARIO == 2:
    print("Toy-Example Data")
    days = 30
    test_days = 30
    temp_load = [[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4] for day in range(days)]
    demand_energy_data.append(temp_load)
    temp_generation = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] for day in range(days)]
    renewable_energy_data.append(temp_generation)
else:
    if SCENARIO == 3:
        n = 0
    elif SCENARIO == 4:
        n = 2
    print(load_file_list[n])
    days = 365
    test_days = 360
    load = (np.load(f"data/load/{load_file_list[n]}")[0:days]).tolist()
    demand_energy_data.append(load)
    generation = (np.load(f"data/generation/{generation_file_list[0]}")[0:days]).tolist()
    renewable_energy_data.append(generation)

print(len(demand_energy_data),len(demand_energy_data[0]),len(demand_energy_data[0][0]))
print(len(renewable_energy_data),len(renewable_energy_data[0]),len(renewable_energy_data[0][0]))

total_iter = 24*(days+days_padding)
total_iter_nopadding = 24*days

if noise_max > 0:
    noise_stddev = np.linspace(0, noise_max*2, 49)
    noise_pre_gen = np.zeros((2, total_iter, n_horizon+1))
    for i in range(n_horizon+1):
        noise_pre_gen[:, :, i] = np.random.normal(0, noise_stddev[i], (2,total_iter))

action_set = np.arange(0,51)/10 * scaling_factor

L_data, G_data, p_tou_ = [], [], []
for day in range(days + days_padding):
    TOU = summer_TOU if SCENARIO == 1 or 90 <= day < 270 else winter_TOU
    p_tou_ += TOU
    for hour in range(24):
        L_data.append([demand_energy_data[0][day%days][hour]*10])
        G_data.append([renewable_energy_data[0][day%days][hour]*10])

print("load size : {}".format(len(L_data)))
print("generation size : {}".format(len(G_data)))
print("tou size : {}".format(len(p_tou_)))



def cont2disc(u0, x0, L, G, sigma):
    idx = np.nonzero(action_set >= u0[0, 0])[0][0]
    if idx == 0:
        u0_disc = action_set[0]
    else:
        u0_discrete_l = action_set[idx-1]
        u0_discrete_u = action_set[idx]
        if np.linalg.norm(u0[0, 0]-u0_discrete_u) < np.linalg.norm(u0[0, 0]-u0_discrete_l):
            u0_disc = u0_discrete_u
        else:
            if x0[0,0]+G+u0_discrete_l-L >= 0:
                u0_disc = u0_discrete_l
            else:
                u0_disc = u0_discrete_u
    u0[0, 0] = u0_disc
    # C_d disc
    u0[1, 0] = max((1-sigma)*x0[1,0], u0_disc)
    # B disc
    u0[2, 0] = min(B_max, x0[0,0]+G+u0_disc-L)
    return u0

######## sigma (for MPC framework) preparation
sigma_ = np.zeros((total_iter, 1))
sigma_[0] = 1
sigma_[24::24] = 1

######## Creating model
model_type = 'discrete'  # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

B = model.set_variable(var_type='_x', var_name='B')
C_d = model.set_variable(var_type='_x', var_name='C_d')
#
a = model.set_variable(var_type='_u', var_name='a')
v_C_d = model.set_variable(var_type='_u', var_name='v_C_d')
v_B = model.set_variable(var_type='_u', var_name='v_B')
#
sigma = model.set_variable(var_type='_tvp', var_name='sigma')
L = model.set_variable(var_type='_tvp', var_name='L')
G = model.set_variable(var_type='_tvp', var_name='G')
p_tou = model.set_variable(var_type='_tvp', var_name='p_tou')

# B_next = fmin(B_max, B+fmin(G+a-L, B_rate))
B_next = v_B
model.set_rhs('B', B_next)

C_d_next = v_C_d
model.set_rhs('C_d', C_d_next)

model.setup()


######## Configuring controller
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': n_horizon,
    't_step': 1,
    'nlpsol_opts': {'ipopt.max_iter': 3000, 'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time':0}
}
mpc.set_param(**setup_mpc)

lterm = p_tou*a+p_d*sigma*C_d
mterm = p_d*sigma*C_d

## objective function
mpc.set_objective(lterm=lterm, mterm=mterm)

## constraints (linear reformulation)
mpc.bounds['lower','_u', 'a'] = 0
mpc.bounds['upper','_u', 'a'] = C_max

mpc.set_nl_cons('a_cons1', L-G-B-a, ub=0)
mpc.set_nl_cons('a_cons2', -a, ub=0)

mpc.set_nl_cons('v_C_d_cons1', v_B-(B+G+a-L), ub=0)
mpc.set_nl_cons('v_C_d_cons2', v_B, ub=B_max)

mpc.set_nl_cons('v_B_cons1', (1-sigma)*C_d-v_C_d, ub=0)
mpc.set_nl_cons('v_B_cons2', a-v_C_d, ub=0)

tvp_template = mpc.get_tvp_template()

def tvp_fun(t_now):
    t_now_int = int(t_now)
    for k in range(n_horizon + 1):
        if t_now_int+k < total_iter:
            if noise_max > 0:
                tvp_template['_tvp', k, 'L'] = max(L_data[t_now_int+k] + noise_pre_gen[0, t_now_int, k],0)
                tvp_template['_tvp', k, 'G'] = max(G_data[t_now_int+k] + noise_pre_gen[1, t_now_int, k],0)
            else:
                tvp_template['_tvp', k, 'L'] = L_data[t_now_int+k]
                tvp_template['_tvp', k, 'G'] = G_data[t_now_int+k]
            tvp_template['_tvp', k, 'p_tou'] = p_tou_[t_now_int+k]
            tvp_template['_tvp', k, 'sigma'] = sigma_[t_now_int+k]

    return tvp_template

mpc.set_tvp_fun(tvp_fun)
mpc.setup()

######## Configuring simulator
simulator = do_mpc.simulator.Simulator(model)
tvp_template_sim = simulator.get_tvp_template()

def tvp_fun_sim(t_now):
    t_now_int = int(t_now)
    tvp_template_sim['L'] = L_data[t_now_int]
    tvp_template_sim['G'] = G_data[t_now_int]
    tvp_template_sim['p_tou'] = p_tou_[t_now_int]
    tvp_template_sim['sigma'] = sigma_[t_now_int]

    return tvp_template_sim

simulator.set_tvp_fun(tvp_fun_sim)
simulator.set_param(t_step=1)
simulator.setup()

######## Results
output = {}
output['a'] = np.zeros((total_iter, 1))
output['B'] = np.zeros((total_iter, 1))
output['p_tou'] = np.zeros((total_iter, 1))
output['p_d'] = np.zeros((total_iter, 1))

######## Creating control loop
x0 = np.array([B_init, 0]).reshape(-1, 1)

simulator.x0 = x0
mpc.x0 = x0

mpc.set_initial_guess()

for k in trange(total_iter):
    u0 = mpc.make_step(x0)
    u0 = cont2disc(u0, x0, L_data[k], G_data[k], sigma_[k])
    x0 = simulator.make_step(u0)
    output['a'][k] = u0[0,0] / scaling_factor
    output['B'][k] = x0[0,0] / scaling_factor
    output['p_tou'][k] = output['p_tou'][k] + p_tou_[k]*output['a'][k] / scaling_factor
    if k % 24 == 23:
        output['p_d'][k] = output['p_d'][k] + p_d * max(output['a'][k-23:k+1]) / scaling_factor

output['total'] = output['p_tou'] + output['p_d']
sim_horizon = np.arange(0,24*days)
print(f"TOU {np.sum(output['p_tou'][sim_horizon])} / DC {np.sum(output['p_d'][sim_horizon])} / total {np.sum(output['total'][sim_horizon])}")

actions=[]
for i in range(days):
    day_action = []
    for j in range(24):
        day_action.append(int(output['a'][i*24+j][0]*10))
    actions.append(day_action)

# calculate cost
TOU_cost = 0
DC_cost = 0
temp_action = []
for day in range(test_days):
    TOU = summer_TOU if SCENARIO == 1 or 90 <= day < 270 else winter_TOU
    day_action = []
    for t in range(24):
        action = actions[day][t]
        day_action.append(action)
        TOU_cost += TOU[t] * action
    DC_cost += max(day_action) * 30
    temp_action.append(day_action)

print("Total_Cost : {}, 평균 : {}".format(TOU_cost+DC_cost,(TOU_cost+DC_cost)//(test_days//30)))
print("TOU_Cost : {}, 평균 : {}".format(TOU_cost,TOU_cost//(test_days//30)))
print("DC_Cost : {}, 평균 : {}".format(DC_cost,DC_cost//(test_days//30)))
print(temp_action)