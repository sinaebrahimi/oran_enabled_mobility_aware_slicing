# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import yaml
# -----import from other files-----------
# for printing styles (e.g., bold, different colors, etc.)
from style import style, convert_seconds
from plot_assistant import plot_graph, moving_average
from initialization import Specifications, Capacity
from radio_calc import Location, RateCalculation
from e2e_calc import Mapping, Delay, StateCalculation
from sac_torch import Agent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# or "1"; change the GPU for multiple simulations (We have 0 and 1 in K80 (zeus401 and zeus402))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ------Loading the parameters-----------
# Define the path to the configuration file
config_file = 'config.yaml'

# Load the configuration file
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Access the values from the config dictionary
USER_NO = config['USER_NO']
BW = eval(config['BW'])
PRB_NO = config['PRB_NO']
# It previously identified it as str! casting to float did not work
SIGMA_NOISE = eval(config['SIGMA_NOISE'])
ETA_AREA = config['ETA_AREA']
X_LIM = config['X_LIM']
# Y_LIM = config['Y_LIM'] # Commented as it is equal to X_LIM
# VM_NO = config['VM_NO']
# VNF_NO = config['VNF_NO']
# NODE_NO = config['NODE_NO']
RU_PER_DU_NO = config['RU_PER_DU_NO']
DU_NO = config['DU_NO']
#BS_NO = config['BS_NO']
BS_NO = RU_PER_DU_NO * DU_NO # 3*2=6
FH_BW_CAPACITY = config['FH_BW_CAPACITY']
E2_BW_CAPACITY = config['E2_BW_CAPACITY']
SLICE_NO = config['SLICE_NO']
MAX_POWER = config['MAX_POWER']
OMEGA_1 = config['OMEGA_1']
VELOCITY = config['VELOCITY']
CONST_D_MAX = config['CONST_D_MAX']
CONST_R_MIN = config['CONST_R_MIN']
PACKET_SIZE = config['PACKET_SIZE']
RAYLEIGH_SCALE = config['RAYLEIGH_SCALE']
T = config['T']
MC = config['MC']
# DRL Hyperparameters
ALPHA_ACT = config['ALPHA_ACT']
BETA_ACT = config['BETA_ACT']
VAR = config['VAR']
DECAY_VAR = config['DECAY_VAR']
# ----------------------------------------
# %
# %% main class


class _main_:
    def __init__(self, MC, T):
        SP = Specifications(USER_NO, SLICE_NO, CONST_D_MAX, CONST_R_MIN, PACKET_SIZE)
        self.mat_specs = SP._()
        # --------------------------------------
        self.loc_user = np.zeros([T, USER_NO, 2])
        # ---------
        CAP = Capacity(RU_PER_DU_NO, DU_NO, BS_NO, FH_BW_CAPACITY, E2_BW_CAPACITY)
        self.mat_links_capacity = CAP.links_capacity()
        # -----------------------------------------------
        # self.mat_reward = np.zeros([T])
        self.mat_reward = -100 * np.ones([MC, T])
        self.mat_satisfied_prb_constraint = np.zeros([MC, T])
        self.mat_satisfied_power_constraint = np.zeros([MC, T])
        self.mat_satisfied_delay_constraint = np.zeros([MC, T])
        self.mat_ssl_u_rate = np.zeros([MC, USER_NO, T])
        self.mat_ssl_u_delay = np.zeros([MC, USER_NO, T])
        self.mat_ssl_rate = np.zeros([MC, T])
        self.mat_ssl_delay = np.zeros([MC, T])
        self.mat_ssl = np.zeros([MC, T])
        self.mat_episode_runtime = np.zeros([MC, T])

        self.shannon = np.zeros([MC, USER_NO, T])

        self.mat_power = np.zeros([MC, USER_NO, T])
        self.mat_gain = np.zeros([MC, USER_NO, PRB_NO, T])
        self.mat_rho  = np.zeros([MC, BS_NO, PRB_NO, USER_NO, T])
        self.mat_u_bs_dist = np.zeros([MC, USER_NO, T])
        # ----------------------
        self.mat_reward_pred = -100 * np.ones([MC, T])
        self.mat_satisfied_prb_constraint_pred = np.zeros([MC, T])
        self.mat_satisfied_power_constraint_pred = np.zeros([MC, T])
        self.mat_satisfied_delay_constraint_pred = np.zeros([MC, T])
        self.mat_ssl_u_rate_pred = np.zeros([MC, USER_NO, T])
        self.mat_ssl_u_delay_pred = np.zeros([MC, USER_NO, T])
        self.mat_ssl_rate_pred = np.zeros([MC, T])
        self.mat_ssl_delay_pred = np.zeros([MC, T])
        self.mat_ssl_pred = np.zeros([MC, T])

        self.shannon_pred = np.zeros([MC, USER_NO, T])

        self.mat_power_pred = np.zeros([MC, USER_NO, T])
        self.mat_gain_pred = np.zeros([MC, USER_NO, PRB_NO, T])
        self.mat_u_bs_dist_pred = np.zeros([MC, USER_NO, T])
        # ---------------------------------------------------------
        #self.monte_mat_rate = np.zeros([MC, T, USER_NO])
        #self.list_rate = []
        # ---------------------------------------------------------
        self.monte_mat_delay_tot = np.zeros([MC, USER_NO, T])
#        self.list_delay = []
        # ---------
        #self.monte_mat_rate_pred = np.zeros([MC, T, USER_NO])
        #self.list_rate_pred = []
        # ---------------------------------------------------------
        self.monte_mat_delay_tot_pred = np.zeros([MC, USER_NO, T])
        # ----------obtaining the number of actions--------------
        self.e1 = BS_NO * PRB_NO * USER_NO
        self.e2 = self.e1 + BS_NO * PRB_NO * USER_NO
        self.e3 = self.e2 + USER_NO * VNF_NO * NODE_NO * VM_NO
        self.e4 = self.e3 + USER_NO
        self.num_actions = self.e4
        # ---------------------------------------------------------
        self.state_size = (2 * USER_NO) + (USER_NO * BS_NO * PRB_NO)  # How is it calculated?

        # calling the SAC agent
        # Calling the SAC agent from sac_torch.py


    def _(self):
        for m in range(MC):
            self.agent = Agent(ALPHA_ACT, BETA_ACT,
                       self.num_actions, self.state_size)
            self.var = VAR
        # .9995 #experiment .9995 and .995 # can determine the ratio of exploration to exploitation
            self.decay_var = DECAY_VAR
            for t in range(T-1):
                start_time = time.time()
                self.tt = t + 1
                if np.mod(t, 500) == 0:
                    print(style.RED + str(t))
                # starting with a negative award, aiming to learn more in initial episodes; Not sure if it is necessary (due to line 495)
                self.reward = -100
                # -----------------------------------------------
                LC = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY,
                              X_LIM, RAYLEIGH_SCALE, ETA_AREA)
                self.loc_user, self.loc_user_pred, self.H, self.H_pred, self.associator, self.associator_pred, self.mat_distance, self.mat_distance_pred = LC.user_location(
                    t, self.loc_user)

                for u in range(USER_NO):
                    for b in range(BS_NO):
                        if self.associator[u, b] == 1:
                            self.mat_u_bs_dist[m, u, t] = self.mat_distance[b, u]
                            self.mat_u_bs_dist_pred[m, u, t] = self.mat_distance_pred[b, u]

                            self.mat_gain[m, u, :, t] = self.H[b, :, u]  # b,k,u
                            self.mat_gain_pred[m, u, :, t] = (self.H_pred[b, :, u])  # b,k,u

                # -----------------------------------------------------
                SC = StateCalculation(self.H, self.loc_user[t, :])
                self.state = SC._()
                self.mat_delay_tot=np.ones([USER_NO])
                self.mat_delay_tot_pred=np.ones([USER_NO])
                self.mat_rate=np.zeros([USER_NO])
                self.mat_rate_pred=np.zeros([USER_NO])
                # -----------------------------------------------------
                self.var = self.var * self.decay_var
                self.noise = np.random.randn(self.num_actions)
                self.noise = self.noise * self.var
                self.action = self.agent.choose_action(self.state)  # Choosing the action
                self.action += self.noise
                self.action = np.clip(self.action, -1, 1)
                # print(self.action) ################
                # -----------------------------
                # -------Current state calculation---------------------
                MA = Mapping(self.action, self.mat_links_capacity, self.mat_nodes_and_vms_capacity, self.mat_specs, self.associator, USER_NO, VM_NO, VNF_NO, NODE_NO, BS_NO, PRB_NO, MAX_POWER)
                self.done_user_prb_allocation, self.rho = MA.ran_prb_allocation()
                if self.done_user_prb_allocation == 1:
                    self.mat_rho[m, :, :, :, t] = self.rho #saving rho
                    self.mat_satisfied_prb_constraint[m, t] = 1
                    self.done_user_power_allocation, self.P = MA.ran_power_allocation()
                    for u in range(USER_NO):
                        self.mat_power[m, u, t] = np.sum(self.P[:, :, u])
                    if self.done_user_power_allocation == 1:
                        self.mat_satisfied_power_constraint[m, t] = 1
                        RC = RateCalculation(self.P, self.rho, self.H, self.associator, BS_NO, PRB_NO, USER_NO, SIGMA_NOISE, BW)
                        self.mat_rate, self.mat_rate_prb = RC._()
                        self.shannon[m, :, t] = self.mat_rate  # b,k,u
                        # -------------------------------------
                        self.W_link = 0  # Why 0? It gives error #obtained from cn_routing()
                        D = Delay(self.mat_rate, self.mat_placement, self.W_link, self.mat_links_capacity, self.mat_nodes_and_vms_capacity, self.mat_specs, self.path, self.associator, self.mat_distance, USER_NO, VNF_NO, BS_NO)
                        done_delay_all,  self.mat_delay_tot = D._()
                        self.monte_mat_delay_tot[m, :, t] = self.mat_delay_tot
                        # -------------------------------------
                        done_delay_dummy = 1  # just tweaking. to not comment the next line
                        if done_delay_dummy == 1:
                            if done_delay_all == 1:
                                self.mat_satisfied_delay_constraint[m, t] = 1
                            self.sigma_SSL_R = 0
                            for s in range(SLICE_NO):
                                for u in range(USER_NO):
                                    if self.mat_specs[u, 0] == s:
                                        # min_rate specification
                                        self.R_s = self.mat_specs[u, 1]
                                        self.mat_ssl_u_rate[m, u, t] = (self.mat_rate[u] / self.R_s)
                                        self.sigma_SSL_R += self.mat_ssl_u_rate[m, u, t]

                            self.SSL_R = self.sigma_SSL_R / (1 + self.sigma_SSL_R)
                            self.mat_ssl_rate[m, t] = self.SSL_R

                            self.sigma_SSL_D = 0
                            for s in range(SLICE_NO):
                                for u in range(USER_NO):
                                    if self.mat_specs[u, 0] == s:
                                        # max_tolerable_delay specification
                                        self.D_s = self.mat_specs[u, 2]
                                        self.mat_ssl_u_delay[m, u, t] = (self.D_s / self.mat_delay_tot[u])
                                        self.sigma_SSL_D += self.mat_ssl_u_delay[m, u, t]

                            self.SSL_D = self.sigma_SSL_D / (1 + self.sigma_SSL_D)
                            self.mat_ssl_delay[m, t] = self.SSL_D
                            # -------------------
                            self.mat_ssl[m, t] = (self.SSL_R**(OMEGA_1)) * ((self.SSL_D)**(1 - OMEGA_1)) # utility function

                            if self.mat_ssl[m, t] >= 0.5:
                                # C10 constraint
                                self.reward += 100 * self.mat_ssl[m, t] # between 0 and 100
                                print(
                                    style.GREEN + 'Reward: {} in episode {} MC {}'.format(self.reward, t, m))
                                self.mat_reward[m, t] = self.reward
                            #else:
                            #    self.reward = 0

                #########################################
                # ------Proactive calculation--------
                # -----------------------------------------------------
                self.reward_pred = -100
                # loc_user is calculated inside LC.user_location (H_pred is its result)
                SC_pred = StateCalculation(self.H_pred, self.loc_user_pred[t, :])
                self.state_pred = SC_pred._()

                ###
                self.action_pred = self.agent.choose_action(
                    self.state_pred)  # Choosing the action
                self.action_pred += self.noise
                self.action_pred = np.clip(self.action_pred, -1, 1)
                ##################
                MA_pred = Mapping(self.action_pred, self.mat_links_capacity, self.mat_nodes_and_vms_capacity,
                                  self.mat_specs, self.associator_pred, USER_NO, VM_NO, VNF_NO, NODE_NO, BS_NO, PRB_NO, MAX_POWER)
                self.done_user_prb_allocation_pred, self.rho_pred = MA_pred.ran_prb_allocation()
                if self.done_user_prb_allocation_pred == 1:
                    self.mat_satisfied_prb_constraint_pred[m, t] = 1
                    self.done_user_power_allocation_pred, self.P_pred = MA_pred.ran_power_allocation()
                    for u in range(USER_NO):
                        self.mat_power_pred[m, u, t] = np.sum(self.P_pred[:, :, u])
                    if self.done_user_power_allocation_pred == 1:
                        self.mat_satisfied_power_constraint_pred[m, t] = 1
                        RC_pred = RateCalculation(self.P_pred, self.rho_pred, self.H_pred, self.associator_pred, BS_NO, PRB_NO, USER_NO, SIGMA_NOISE, BW)
                        self.mat_rate_pred, self.mat_rate_prb_pred = RC_pred._() # m, u, t

                        self.shannon_pred[m, :, t] = self.mat_rate_pred
                        # -------------------------------------
                        self.W_link_pred = 0  # Why 0? It gives error #obtained from cn_routing()
                        D_pred = Delay(self.mat_rate_pred, self.mat_placement_pred, self.W_link_pred, self.mat_links_capacity, self.mat_nodes_and_vms_capacity,
                                        self.mat_specs, self.path_pred, self.associator_pred, self.mat_distance_pred, USER_NO, VNF_NO, BS_NO)
                        done_delay_all_pred,  self.mat_delay_tot_pred = D_pred._()
                        # -------------------------------------
                        self.monte_mat_delay_tot_pred[m, :, t] = self.mat_delay_tot_pred
                        done_delay_dummy = 1  # just tweaking. to not comment the next line
                        if done_delay_dummy == 1:
                            if done_delay_all_pred == 1:
                                self.mat_satisfied_delay_constraint_pred[m, t] = 1
                            self.sigma_SSL_R_pred = 0
                            for s in range(SLICE_NO):
                                for u in range(USER_NO):
                                    if self.mat_specs[u, 0] == s:
                                        # min_rate specification
                                        self.R_s = self.mat_specs[u, 1]
                                        self.mat_ssl_u_rate_pred[m, u, t] = (self.mat_rate_pred[u] / self.R_s)
                                        self.sigma_SSL_R_pred += self.mat_ssl_u_rate_pred[m, u, t]

                            self.SSL_R_pred = self.sigma_SSL_R_pred / (1 + self.sigma_SSL_R_pred)
                            self.mat_ssl_rate_pred[m, t] = self.SSL_R_pred

                            self.sigma_SSL_D_pred = 0
                            for s in range(SLICE_NO):
                                for u in range(USER_NO):
                                    if self.mat_specs[u, 0] == s:
                                        # max_tolerable_delay specification
                                        self.D_s = self.mat_specs[u, 2]
                                        self.mat_ssl_u_delay_pred[m, u, t] = (self.D_s / self.mat_delay_tot_pred[u])
                                        self.sigma_SSL_D_pred += self.mat_ssl_u_delay_pred[m, u, t]

                            self.SSL_D_pred = self.sigma_SSL_D_pred / (1 + self.sigma_SSL_D_pred)
                            self.mat_ssl_delay_pred[m, t] = self.SSL_D_pred
                            
                            # -------------------
                            self.mat_ssl_pred[m, t] = (self.SSL_R_pred**(OMEGA_1)) * ((self.SSL_D_pred)**(1 - OMEGA_1))

                            if self.mat_ssl_pred[m, t] >= 0.5:
                                # C10 constraint
                                self.reward_pred += 100 * self.mat_ssl_pred[m, t]
                                print(style.BLUE + 'Reward (Proactive): {} in episode {} MC {}'.format(self.reward_pred, t, m))
                                self.mat_reward_pred[m, t] = self.reward_pred

                ##############################
                # ---------Next state calculation--------------
                LC = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY,
                              X_LIM, RAYLEIGH_SCALE, ETA_AREA)
                self.loc_users_new, self.loc_users_new_pred, self.H_new, self.H_pred, self.associator, self.associator_pred, self.mat_distance, self.mat_distance_pred = LC.user_location(
                    self.tt, self.loc_user)
                # -----------------------------------------------------
                SC = StateCalculation(self.H_new, self.loc_users_new[t, :])
                self.next_state = SC._()
                self.next_state = self.next_state.astype('float16')
                # -------------------------------------
                self.agent.memorize(self.state, self.action,
                                    self.reward, self.next_state)
                self.agent.replay()
                # ---------------------------------------------------------
                # self.list_delay.append(np.sum(self.mat_delay_tot) * 1000) # delay in ms
                # self.list_rate.append(np.sum(self.mat_rate)) # rate in Mbps
    #            self.list_delay.append((np.sum(self.mat_delay_tot) / USER_NO)) # average delay in ms
    #            self.list_rate.append(np.sum(self.mat_rate) / USER_NO) # average rate in Mbps
                # average delay in ms
                #self.list_delay.append(np.average(self.mat_delay_tot))
                # average rate in Mbps
                #self.list_rate.append(np.average(self.mat_rate))

                #self.list_delay_pred.append(np.average(self.mat_delay_tot_pred))  # average delay in ms
                #self.list_rate_pred.append(np.average(self.mat_rate_pred))  # average rate in Mbps

                end_time = time.time()  # Record the end time of the loop
                # Storing the episode/timeslot runtime duration in seconds
                self.mat_episode_runtime[m,t] = end_time - start_time

        return self.mat_rho, self.mat_u_bs_dist, self.mat_u_bs_dist_pred, self.shannon, self.shannon_pred, self.mat_gain, self.mat_gain_pred, self.mat_power, self.mat_power_pred, self.mat_reward, self.mat_reward_pred, self.mat_satisfied_prb_constraint, self.mat_satisfied_prb_constraint_pred, self.mat_satisfied_power_constraint, self.mat_satisfied_power_constraint_pred, self.mat_satisfied_delay_constraint, self.mat_satisfied_delay_constraint_pred, self.mat_ssl_rate, self.mat_ssl_rate_pred, self.mat_ssl_delay, self.mat_ssl_delay_pred, self.mat_ssl, self.mat_ssl_pred, self.mat_episode_runtime, self.mat_rate, self.mat_rate_pred, self.mat_delay_tot, self.mat_delay_tot_pred


# %%
M = _main_(MC, T)
mat_rho, mat_u_bs_dist, mat_u_bs_dist_pred, shannon, shannon_pred, mat_gain, mat_gain_pred, mat_power, mat_power_pred, mat_reward, mat_reward_pred, mat_satisfied_prb_constraint, mat_satisfied_prb_constraint_pred, mat_satisfied_power_constraint, mat_satisfied_power_constraint_pred, mat_satisfied_delay_constraint, mat_satisfied_delay_constraint_pred, mat_ssl_rate, mat_ssl_rate_pred, mat_ssl_delay, mat_ssl_delay_pred, mat_ssl, mat_ssl_pred, mat_episode_runtime, mat_rate, mat_rate_pred, mat_delay_tot, mat_delay_tot_pred = M._()

# %PLOTTING THE RESULTS%%
# shannon_0 = np.sum(shannon , axis=0)
# variance_shannon_over_t = np.std(shannon_0) #var on all users
# variance for all episodes and users (matrix)
variance_shannon = np.std(shannon)
mem_shannon_t = []
mem_shannon_t_min = []
mem_shannon_t_avg = []
mem_shannon_t_max = []

mem_shannon_u = []
mem_shannon_u_min = []
mem_shannon_u_avg = []
mem_shannon_u_max = []



#for t in range(T):
#    mem_shannon_t.append(np.std(shannon[:, :, t])) # m, u, t
#    mem_shannon_t_min.append(np.min(shannon[:, :, t]))
#    mem_shannon_t_avg.append(np.average(shannon[:, :, t]))
#    mem_shannon_t_max.append(np.max(shannon[:, :, t]))

for u in range(USER_NO):
    mem_shannon_u.append(np.std(shannon[:, u, :]))
    mem_shannon_u_min.append(np.min(shannon[:, u, :]))
    mem_shannon_u_avg.append(np.average(shannon[:, u, :]))
    mem_shannon_u_max.append(np.max(shannon[:, u, :]))


# Define moving average function
# def moving_average(data, window_size):
#     cumsum_vec = np.cumsum(np.insert(data, 0, 0))
#     ma_vec = (cumsum_vec[window_size:] -
#               cumsum_vec[:-window_size]) / window_size
#     return ma_vec


window_size = 50
# Calculate moving average with specified window size
#mem_shannon_t_smooth = moving_average(mem_shannon_t, window_size)
#mem_shannon_t_min_smooth = moving_average(mem_shannon_t_min, window_size)
#mem_shannon_t_avg_smooth = moving_average(mem_shannon_t_avg, window_size)
#mem_shannon_t_max_smooth = moving_average(mem_shannon_t_max, window_size)

# Plot smoothed data
#plt.plot(mem_shannon_t_smooth, label='Variance')
#plt.plot(mem_shannon_t_min_smooth, label='min')
#plt.plot(mem_shannon_t_avg_smooth, label='avg')
#plt.plot(mem_shannon_t_max_smooth, label='max')
#plt.xlabel("Episode")
#plt.ylabel("Data rate (Mbps) (Shannon Formula)")
#plt.legend()
#plt.show()

plt.plot(mem_shannon_u, label='Variance')
plt.plot(mem_shannon_u_min, label='min')
plt.plot(mem_shannon_u_avg, label='avg')
plt.plot(mem_shannon_u_max, label='max')
plt.legend()
plt.xlabel("User #")
plt.ylabel("Data rate (Mbps) (Shannon Formula)")
plt.show()

print("\===============Shannon Variance for all users and episodes (matrix)==================")
print("Shannon Variance:", variance_shannon)


#########
#
# variance for all episodes and users (matrix)
variance_mat_u_bs_dist = np.std(mat_u_bs_dist)
mem_mat_u_bs_dist_t = []
mem_mat_u_bs_dist_t_min = []
mem_mat_u_bs_dist_t_avg = []
mem_mat_u_bs_dist_t_max = []

mem_mat_u_bs_dist_u = []
mem_mat_u_bs_dist_u_min = []
mem_mat_u_bs_dist_u_avg = []
mem_mat_u_bs_dist_u_max = []
for t in range(T):
    mem_mat_u_bs_dist_t.append(np.std(mat_u_bs_dist[:, :, t]))
    mem_mat_u_bs_dist_t_min.append(np.min(mat_u_bs_dist[:, :, t]))
    mem_mat_u_bs_dist_t_avg.append(np.average(mat_u_bs_dist[:, :, t]))
    mem_mat_u_bs_dist_t_max.append(np.max(mat_u_bs_dist[:, :, t]))

for u in range(USER_NO):
    mem_mat_u_bs_dist_u.append(np.std(mat_u_bs_dist[:, u, :]))
    mem_mat_u_bs_dist_u_min.append(np.min(mat_u_bs_dist[:, u, :]))
    mem_mat_u_bs_dist_u_avg.append(np.average(mat_u_bs_dist[:, u, :]))
    mem_mat_u_bs_dist_u_max.append(np.max(mat_u_bs_dist[:, u, :]))

# Calculate moving average with specified window size
mem_mat_u_bs_dist_t_smooth = moving_average(mem_mat_u_bs_dist_t, window_size)
mem_mat_u_bs_dist_t_min_smooth = moving_average(
    mem_mat_u_bs_dist_t_min, window_size)
mem_mat_u_bs_dist_t_avg_smooth = moving_average(
    mem_mat_u_bs_dist_t_avg, window_size)
mem_mat_u_bs_dist_t_max_smooth = moving_average(
    mem_mat_u_bs_dist_t_max, window_size)

# Plot smoothed data
print('Distance of users with the selcted BS (analysis on episodes):\n')
plt.plot(mem_mat_u_bs_dist_t_smooth, label='Variance')
plt.plot(mem_mat_u_bs_dist_t_min_smooth, label='min')
plt.plot(mem_mat_u_bs_dist_t_avg_smooth, label='avg')
plt.plot(mem_mat_u_bs_dist_t_max_smooth, label='max')
plt.xlabel("Episode")
plt.ylabel("Distance (m)")
plt.legend()
plt.show()

print('Distance of users with the selcted BS (analysis on users):\n')

plt.plot(mem_mat_u_bs_dist_u, label='Variance')
plt.plot(mem_mat_u_bs_dist_u_min, label='min')
plt.plot(mem_mat_u_bs_dist_u_avg, label='avg')
plt.plot(mem_mat_u_bs_dist_u_max, label='max')
plt.legend()
plt.xlabel("User #")
plt.ylabel("Distance (m)")
plt.show()

print("\===============mat_u_bs_dist Variance for all users and episodes (matrix)==================")
print("mat_u_bs_dist Variance:", variance_mat_u_bs_dist)


######

avg_mat_u_bs_dist = np.average(mat_u_bs_dist, axis=(0,1))
max_mat_u_bs_dist = np.max(mat_u_bs_dist, axis=(0,1))
min_mat_u_bs_dist = np.min(mat_u_bs_dist, axis=(0,1))

avg_mat_u_bs_dist_pred = np.average(mat_u_bs_dist_pred, axis=(0,1))
max_mat_u_bs_dist_pred = np.max(mat_u_bs_dist_pred, axis=(0,1))
min_mat_u_bs_dist_pred = np.min(mat_u_bs_dist_pred, axis=(0,1))
# Compute the moving average for each data series
min_mat_u_bs_dist_smooth = np.convolve(
    min_mat_u_bs_dist, np.ones(window_size)/window_size, mode='same')
max_mat_u_bs_dist_smooth = np.convolve(
    max_mat_u_bs_dist, np.ones(window_size)/window_size, mode='same')
avg_mat_u_bs_dist_smooth = np.convolve(
    avg_mat_u_bs_dist, np.ones(window_size)/window_size, mode='same')

# sum_mat_u_bs_dist_pred_smooth = np.convolve(
#     sum_mat_u_bs_dist_pred, np.ones(window_size)/window_size, mode='same')
#-----------------------------------------------------------------------------------------------
min_mat_u_bs_dist_pred_smooth = np.convolve(
    min_mat_u_bs_dist_pred, np.ones(window_size)/window_size, mode='same')
max_mat_u_bs_dist_pred_smooth = np.convolve(
    max_mat_u_bs_dist_pred, np.ones(window_size)/window_size, mode='same')
avg_mat_u_bs_dist_pred_smooth = np.convolve(
    avg_mat_u_bs_dist_pred, np.ones(window_size)/window_size, mode='same')

# Create a figure and axis
fig, ax = plt.subplots()

# plt.plot(mat_u_bs_dist[0,:])
# plt.plot(mat_u_bs_dist[1,:])
# plt.plot(mat_u_bs_dist[2,:])
# plt.plot(mat_u_bs_dist[3,:])
# plt.plot(mat_u_bs_dist[4,:])
# plt.show()

# Plot the smoothed values with different colors and a solid line style
# ax.plot(sum_mat_u_bs_dist_smooth, color='blue', linestyle='solid', label='Sum')
ax.plot(min_mat_u_bs_dist_smooth, color='green',
        linestyle='solid', label='Min')
ax.plot(max_mat_u_bs_dist_smooth, color='red', linestyle='solid', label='Max')
ax.plot(avg_mat_u_bs_dist_smooth, color='orange',
        linestyle='solid', label='Avg')

# Plot the smoothed _pred values with different colors and a dotted line style
# ax.plot(sum_mat_u_bs_dist_pred_smooth, color='blue', linestyle='dotted', label='Sum (Pred)')
ax.plot(min_mat_u_bs_dist_pred_smooth, color='green',
        linestyle='dotted', label='Min (Pred)')
ax.plot(max_mat_u_bs_dist_pred_smooth, color='red',
        linestyle='dotted', label='Max (Pred)')
ax.plot(avg_mat_u_bs_dist_pred_smooth, color='orange',
        linestyle='dotted', label='Avg (Pred)')

# Set the x and y labels
ax.set_xlabel("Episode")
ax.set_ylabel("mat_u_bs_dist of users x(m)")

# Add a legend
ax.legend()

# Show the plot
plt.show()
#####
# shannon (rate)
# rate:
window_size = 50
sum_rate = np.sum(shannon, axis=(0,1))
std_rate = np.std(shannon, axis=(0,1))
avg_rate = np.average(shannon, axis=(0,1))
max_rate = np.max(shannon, axis=(0,1))
min_rate = np.min(shannon, axis=(0,1))

sum_rate_pred = np.sum(shannon_pred, axis=(0,1))
std_rate_pred = np.std(shannon_pred, axis=(0,1))
avg_rate_pred = np.average(shannon_pred, axis=(0,1))
max_rate_pred = np.max(shannon_pred, axis=(0,1))
min_rate_pred = np.min(shannon_pred, axis=(0,1))
# Compute the moving average for each data series
sum_rate_smooth = np.convolve(sum_rate, np.ones(
    window_size)/window_size, mode='same')
std_rate_smooth = np.convolve(std_rate, np.ones(
    window_size)/window_size, mode='same')
min_rate_smooth = np.convolve(min_rate, np.ones(
    window_size)/window_size, mode='same')
max_rate_smooth = np.convolve(max_rate, np.ones(
    window_size)/window_size, mode='same')
avg_rate_smooth = np.convolve(avg_rate, np.ones(
    window_size)/window_size, mode='same')

sum_rate_pred_smooth = np.convolve(
    sum_rate_pred, np.ones(window_size)/window_size, mode='same')
std_rate_pred_smooth = np.convolve(
    std_rate_pred, np.ones(window_size)/window_size, mode='same')
min_rate_pred_smooth = np.convolve(
    min_rate_pred, np.ones(window_size)/window_size, mode='same')
max_rate_pred_smooth = np.convolve(
    max_rate_pred, np.ones(window_size)/window_size, mode='same')
avg_rate_pred_smooth = np.convolve(
    avg_rate_pred, np.ones(window_size)/window_size, mode='same')

# Create a figure and axis
fig, ax = plt.subplots()

# plt.plot(shannon[0,:])
# plt.plot(shannon[1,:])
# plt.plot(shannon[2,:])
# plt.plot(shannon[3,:])
# plt.plot(shannon[4,:])
# plt.show()

# Plot the smoothed values with different colors and a solid line style
# ax.plot(sum_rate_smooth, color='blue', linestyle='solid', label='Sum')
ax.plot(std_rate_smooth, color='blue', linestyle='solid', label='Variance')
ax.plot(min_rate_smooth, color='green', linestyle='solid', label='Min')
ax.plot(max_rate_smooth, color='red', linestyle='solid', label='Max')
ax.plot(avg_rate_smooth, color='orange', linestyle='solid', label='Avg')

# Plot the smoothed _pred values with different colors and a dotted line style
# ax.plot(sum_rate_pred_smooth, color='blue', linestyle='dotted', label='Sum (Pred)')
ax.plot(std_rate_pred_smooth, color='blue',
        linestyle='dotted', label='Variance (Pred)')
ax.plot(min_rate_pred_smooth, color='green',
        linestyle='dotted', label='Min (Pred)')
ax.plot(max_rate_pred_smooth, color='red',
        linestyle='dotted', label='Max (Pred)')
ax.plot(avg_rate_pred_smooth, color='orange',
        linestyle='dotted', label='Avg (Pred)')

# Set the x and y labels
ax.set_xlabel("Episode")
ax.set_ylabel("Rate of users (Mbps)")

# Add a legend
ax.legend()

# Show the plot
plt.show()

####
fig, ax1 = plt.subplots()

ax1.plot(sum_rate_smooth, color='blue', linestyle='solid', label='Sum')
ax1.plot(sum_rate_pred_smooth, color='blue',
         linestyle='dotted', label='Sum (Pred)')
# Set the x and y labels
ax1.set_xlabel("Episode")
ax1.set_ylabel("sum rate of users")

# Add a legend
ax1.legend()

# Show the plot
plt.show()


# POWER:
window_size = 50
sum_power = np.sum(mat_power, axis=(0,1))
avg_power = np.mean(mat_power, axis=(0,1))
max_power = np.max(mat_power, axis=(0,1))
min_power = np.min(mat_power, axis=(0,1))


sum_power_pred = np.sum(mat_power_pred, axis=(0,1))
avg_power_pred = np.mean(mat_power_pred, axis=(0,1))
max_power_pred = np.max(mat_power_pred, axis=(0,1))
min_power_pred = np.min(mat_power_pred, axis=(0,1))
# Compute the moving average for each data series
sum_power_smooth = np.convolve(sum_power, np.ones(
    window_size)/window_size, mode='same')
min_power_smooth = np.convolve(min_power, np.ones(
    window_size)/window_size, mode='same')
max_power_smooth = np.convolve(max_power, np.ones(
    window_size)/window_size, mode='same')
avg_power_smooth = np.convolve(avg_power, np.ones(
    window_size)/window_size, mode='same')

sum_power_pred_smooth = np.convolve(
    sum_power_pred, np.ones(window_size)/window_size, mode='same')
min_power_pred_smooth = np.convolve(
    min_power_pred, np.ones(window_size)/window_size, mode='same')
max_power_pred_smooth = np.convolve(
    max_power_pred, np.ones(window_size)/window_size, mode='same')
avg_power_pred_smooth = np.convolve(
    avg_power_pred, np.ones(window_size)/window_size, mode='same')

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the smoothed values with different colors and a solid line style
ax.plot(sum_power_smooth, color='blue', linestyle='solid', label='Sum')
ax.plot(min_power_smooth, color='green', linestyle='solid', label='Min')
ax.plot(max_power_smooth, color='red', linestyle='solid', label='Max')
ax.plot(avg_power_smooth, color='orange', linestyle='solid', label='Avg')

# Plot the smoothed _pred values with different colors and a dotted line style
ax.plot(sum_power_pred_smooth, color='blue',
        linestyle='dotted', label='Sum (Pred)')
ax.plot(min_power_pred_smooth, color='green',
        linestyle='dotted', label='Min (Pred)')
ax.plot(max_power_pred_smooth, color='red',
        linestyle='dotted', label='Max (Pred)')
ax.plot(avg_power_pred_smooth, color='orange',
        linestyle='dotted', label='Avg (Pred)')

# Set the x and y labels
ax.set_xlabel("Episode")
ax.set_ylabel("power of users")

# Add a legend
ax.legend()

# Show the plot
plt.show()

# gain:
print('#####################')
print('REMEMBER TO PLOT GAIN')
#gain vector 1*PRB . rho(u,:) = sum gains for the user on that PRB
#rate (gain , power)

# for user=2 # rho(m,b,k,u,t) /// gain(m,u,k,t)
mat_gain_monte_1 = mat_gain[0,:,:,:] # for m=0
mat_gain_user_1_monte_1 = mat_gain_monte_1[0, :, :] #for user 0

mat_rho_monte_1 = mat_rho[0, :,:,:,:]
mat_rho_user_1_monte_1 = mat_rho_monte_1[:,:, 0, :]

list_gain_user_1 = [] # changes in sum(gain) in time
mat_gain_user_1_monte_1 = np.transpose(mat_gain_user_1_monte_1) # to make it multiplicable

for b in range(BS_NO):
    if np.dot(mat_rho_user_1_monte_1[b, :, :], mat_gain_user_1_monte_1) > 0:
        list_gain_user_1.append(np.dot(mat_rho_user_1_monte_1[b, :, :], mat_gain_user_1_monte_1))

plt.plot(list_gain_user_1, label='sum gain for user #1')
plt.xlabel("Episode")
plt.ylabel("sum gain for user #1")
plt.legend()
plt.show()

#sum_gain = np.sum(mat_gain, axis=0) #m,u,k,t
#avg_gain = np.mean(mat_gain, axis=0)
#max_gain = np.max(mat_gain, axis=0)
#min_gain = np.min(mat_gain, axis=0)
#
#
#sum_gain_pred = np.sum(mat_gain_pred, axis=0)
#avg_gain_pred = np.mean(mat_gain_pred, axis=0)
#max_gain_pred = np.max(mat_gain_pred, axis=0)
#min_gain_pred = np.min(mat_gain_pred, axis=0)
## Compute the moving average for each data series
#sum_gain_smooth = np.convolve(sum_gain, np.ones(
#    window_size)/window_size, mode='same')
#min_gain_smooth = np.convolve(min_gain, np.ones(
#    window_size)/window_size, mode='same')
#max_gain_smooth = np.convolve(max_gain, np.ones(
#    window_size)/window_size, mode='same')
#avg_gain_smooth = np.convolve(avg_gain, np.ones(
#    window_size)/window_size, mode='same')
#
#sum_gain_pred_smooth = np.convolve(
#    sum_gain_pred, np.ones(window_size)/window_size, mode='same')
#min_gain_pred_smooth = np.convolve(
#    min_gain_pred, np.ones(window_size)/window_size, mode='same')
#max_gain_pred_smooth = np.convolve(
#    max_gain_pred, np.ones(window_size)/window_size, mode='same')
#avg_gain_pred_smooth = np.convolve(
#    avg_gain_pred, np.ones(window_size)/window_size, mode='same')
#
## Create a figure and axis
#fig, ax = plt.subplots()
#
## Plot the smoothed values with different colors and a solid line style
#ax.plot(sum_gain_smooth, color='blue', linestyle='solid', label='Sum')
#ax.plot(min_gain_smooth, color='green', linestyle='solid', label='Min')
#ax.plot(max_gain_smooth, color='red', linestyle='solid', label='Max')
#ax.plot(avg_gain_smooth, color='orange', linestyle='solid', label='Avg')
#
## Plot the smoothed _pred values with different colors and a dotted line style
#ax.plot(sum_gain_pred_smooth, color='blue',
#        linestyle='dotted', label='Sum (Pred)')
#ax.plot(min_gain_pred_smooth, color='green',
#        linestyle='dotted', label='Min (Pred)')
#ax.plot(max_gain_pred_smooth, color='red',
#        linestyle='dotted', label='Max (Pred)')
#ax.plot(avg_gain_pred_smooth, color='orange',
#        linestyle='dotted', label='Avg (Pred)')
#
## Set the x and y labels
#ax.set_xlabel("Episode")
#ax.set_ylabel("Channel gain")
#
## Add a legend
#ax.legend()
#
## Show the plot
#plt.show()

# %%REWARD%%%%
# ep_rewardall = mat_reward
mat_reward_average_over_m=np.average(mat_reward,axis=0)
mat_reward_average_over_m_pred=np.average(mat_reward_pred,axis=0)
#===========================================================
# aaa = len(ep_rewardall)
# mean_ep_rewardall = []
# mean_ep_rewardall_pred = []
# for i in range(aaa-w):
#     mean_ep_rewardall.append(np.sum(ep_rewardall[i:-aaa + w+i])/w)
#     mean_ep_rewardall_pred.append(np.sum(mat_reward_pred[i:-aaa + w+i])/w)
#===========================================================
window_size = 200  # 50 (for smoothing the curves)
mean_ep_rewardall=moving_average(mat_reward_average_over_m,window_size)
mean_ep_rewardall_pred=moving_average(mat_reward_average_over_m_pred,window_size)
plt.plot(mean_ep_rewardall, label='SAC')
plt.plot(mean_ep_rewardall_pred, label='Proactive SAC')
plt.xlabel("Episode")
plt.ylabel("Mean episodic rewards")
plt.legend()
plt.show()
# %%CONSTRAINT SATISFACTION%
mean_ep_prb_const = moving_average(np.average(mat_satisfied_prb_constraint,axis=0), window_size) #[]
mean_ep_power_const =  moving_average(np.average(mat_satisfied_power_constraint,axis=0), window_size) #[]
mean_ep_delay_const =  moving_average(np.average(mat_satisfied_delay_constraint,axis=0), window_size) #[]
# for i in range(aaa-w):
#     mean_ep_prb_const.append(
#         (100 * np.sum(mat_satisfied_prb_constraint[i:-aaa + w+i]))/w)
#     mean_ep_power_const.append(
#         (100 * np.sum(mat_satisfied_power_constraint[i:-aaa + w+i]))/w)
#     mean_ep_delay_const.append(
#         (100 * np.sum(mat_satisfied_delay_constraint[i:-aaa + w+i]))/w)

    #=======================
plt.plot(mean_ep_prb_const, label='PRB satisfaction: {:.2f}%'.format(
    100 * np.average(mat_satisfied_prb_constraint)))
plt.plot(mean_ep_power_const, label='Power satisfaction: {:.2f}%'.format(
    100 * np.average(mat_satisfied_power_constraint)))
plt.plot(mean_ep_delay_const, label='Delay satisfaction: {:.2f}%'.format(
    100 * np.average(mat_satisfied_delay_constraint)))
plt.xlabel("Episode")
plt.ylabel("Constraint satisfaction (in %) ")
plt.legend()
plt.show()
# %%%%%SSL%%%%%%
mean_ep_ssl_rate =  moving_average(np.average(mat_ssl_rate,axis=0), window_size)  #[]
mean_ep_ssl_delay = moving_average(np.average(mat_ssl_delay,axis=0), window_size)#[]
mean_ep_ssl = moving_average(np.average(mat_ssl,axis=0), window_size)#[]
mean_ep_ssl_rate_pred = moving_average(np.average(mat_ssl_rate_pred,axis=0), window_size)#[]
mean_ep_ssl_delay_pred = moving_average(np.average(mat_ssl_delay_pred,axis=0), window_size)#[]
mean_ep_ssl_pred = moving_average(np.average(mat_ssl_pred,axis=0), window_size)#[]
#==========================================================
# for i in range(aaa-w):
#     mean_ep_ssl_rate.append((np.sum(mat_ssl_rate[i:-aaa + w+i]))/w)
#     mean_ep_ssl_delay.append((np.sum(mat_ssl_delay[i:-aaa + w+i]))/w)
#     mean_ep_ssl.append((np.sum(mat_ssl[i:-aaa + w+i]))/w)
#     mean_ep_ssl_rate_pred.append((np.sum(mat_ssl_rate_pred[i:-aaa + w+i]))/w)
#     mean_ep_ssl_delay_pred.append((np.sum(mat_ssl_delay_pred[i:-aaa + w+i]))/w)
#     mean_ep_ssl_pred.append((np.sum(mat_ssl_pred[i:-aaa + w+i]))/w)


line1, = plt.plot(mean_ep_ssl_rate, label='SSL_1 (Rate satisfaction); Average: {:.2f}'.format(
    np.average(mat_ssl_rate)))
color1 = line1.get_color()  # get its color
plt.plot(mean_ep_ssl_rate_pred, color=color1, linestyle='--',
         label='(Proactive) SSL_1 (Rate satisfaction); Average: {:.2f}'.format(np.average(mat_ssl_rate_pred)))

line2, = plt.plot(mean_ep_ssl_delay, label='SSL_2 (Delay satisfaction); Average: {:.2f}'.format(
    np.average(mat_ssl_delay)))
color2 = line2.get_color()  # get its color
plt.plot(mean_ep_ssl_delay_pred, color=color2, linestyle='--',
         label='(Proactive) SSL_2(Delay satisfaction); Average: {:.2f}'.format(np.average(mat_ssl_delay_pred)))

line3, = plt.plot(mean_ep_ssl, label='SSL (Overall satisfaction); Average: {:.2f}'.format(
    np.average(mat_ssl)))
color3 = line3.get_color()  # get its color
plt.plot(mean_ep_ssl_rate_pred, color=color3, linestyle='--',
         label='(Proactive) SSL (Overall satisfaction); Average: {:.2f}'.format(np.average(mat_ssl_pred)))

# plt.plot(mean_ep_ssl_delay, label='SSL_2 (Delay satisfaction); Average: {:.2f}'.format(np.average(mat_ssl_delay)))
# plt.plot(mean_ep_ssl, label='SSL (Overall satisfaction); Average: {:.2f}'.format(np.average(mat_ssl)))
plt.xlabel("Episode")
plt.ylabel("SSL")
plt.legend()
plt.show()

# %%%%%Delay%%%%%%
mat_delay_tot_average_over_m=np.average(mat_delay_tot, axis=0)#m,u,t
mat_delay_tot_average_over_m_over_u = np.average(mat_delay_tot_average_over_m, axis=0)
mean_mat_delay_tot_average_over_m_over_u = moving_average(mat_delay_tot_average_over_m_over_u, window_size)

mat_delay_tot_pred_average_over_m=np.average(mat_delay_tot_pred, axis=0)#m,u,t
mat_delay_tot_pred_average_over_m_over_u = np.average(mat_delay_tot_pred_average_over_m, axis=0)
mean_mat_delay_tot_pred_average_over_m_over_u = moving_average(mat_delay_tot_pred_average_over_m_over_u, window_size)

plt.plot(mean_mat_delay_tot_average_over_m_over_u, label='SAC')
plt.plot(mean_mat_delay_tot_pred_average_over_m_over_u, label='Proactive')
plt.xlabel("Episode")
plt.ylabel("Average E2E Delay of users (ms)")
plt.legend()
plt.show()
# mean_ep_delay = []
# mean_ep_delay_pred = []
# for i in range(aaa-w):
#     mean_ep_delay.append(np.sum(list_delay[i:-aaa + w+i])/w)
#     mean_ep_delay_pred.append(np.sum(list_delay_pred[i:-aaa + w+i])/w)
# plt.plot(mean_ep_delay, label='SAC')
# plt.plot(mean_ep_delay, label='SAC (Proactive)')
# plt.xlabel("Episode")
# plt.ylabel("Average E2E Delay (ms)")
# plt.legend()
# plt.show()
# %%%%RUNTIME DURATION%%%%%%%
#window_size = 200
mat_episode_runtime_average_over_m=np.average(mat_episode_runtime, axis=0)
mean_mat_episode_runtime = moving_average(mat_reward_average_over_m_pred, window_size)

plt.plot(mean_mat_episode_runtime, label='Average runtime duration: {:.2f} ms'.format(1000 * np.average(mat_episode_runtime)))
plt.xlabel("Episode")
plt.ylabel("Runtime duration (ms)")
plt.legend()
plt.show()

#aaa=T
#ep_time = mat_episode_runtime
#mean_ep_time = []
#for i in range(aaa-window_size):
#    temp_value = np.sum((1000 * ep_time[i:-aaa + window_size+i]))/window_size
#    mean_ep_time.append(temp_value)

print(style.UNDERLINE + "Total time for {} timeslots/episodes ({} users) in {} Monte-Carlo iterations: {}".format(T, USER_NO, MC, convert_seconds(np.sum(mat_episode_runtime))))

# %%%%%%%

# save for later (use savez_compressed for compression)
filename = f'O-RAN SAC (Normal and Proactive), RAYLEIGH={RAYLEIGH_SCALE}, U={USER_NO}, PRB={PRB_NO}, T={T}, SPEED={USER_SPEED}, OMEGA_1={OMEGA_1}, D_max={CONST_D_MAX}, R_min={CONST_R_MIN}.npz'
np.savez_compressed(filename, mat_rho=mat_rho, mat_u_bs_dist=mat_u_bs_dist, mat_u_bs_dist_pred=mat_u_bs_dist_pred, shannon=shannon, shannon_pred=shannon_pred, mat_gain=mat_gain, mat_gain_pred=mat_gain_pred, mat_power=mat_power, mat_power_pred=mat_power_pred, mat_reward=mat_reward, mat_reward_pred=mat_reward_pred, mat_satisfied_prb_constraint=mat_satisfied_prb_constraint, mat_satisfied_prb_constraint_pred=mat_satisfied_prb_constraint_pred, mat_satisfied_power_constraint=mat_satisfied_power_constraint, mat_satisfied_power_constraint_pred=mat_satisfied_power_constraint_pred,
                    mat_satisfied_delay_constraint=mat_satisfied_delay_constraint, mat_satisfied_delay_constraint_pred=mat_satisfied_delay_constraint_pred, mat_ssl_rate=mat_ssl_rate, mat_ssl_rate_pred=mat_ssl_rate_pred, mat_ssl_delay=mat_ssl_delay, mat_ssl_delay_pred=mat_ssl_delay_pred, mat_ssl=mat_ssl, mat_ssl_pred=mat_ssl_pred, mat_episode_runtime=mat_episode_runtime, mat_rate=mat_rate, mat_rate_pred=mat_rate_pred, mat_delay_tot=mat_delay_tot, mat_delay_tot_pred=mat_delay_tot_pred)
