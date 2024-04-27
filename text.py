# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import yaml
# -----import from other files-----------
from style import style, convert_seconds
from plot_assistant import plot_graph, moving_average
from initialization import Specifications
from radio_calc import Location, RateCalculation
from e2e_calc import Mapping, Delay, StateCalculation
from sac_torch import Agent
np.random.seed(1372) # some random number
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ------Loading the parameters-----------
# Define the path to the configuration file
config_file = 'config.yaml'

# Load the configuration file
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Access the values from the config dictionary
USER_NO = config['USER_NO']
BW = config['BW'] # eval(config['BW'])
PRB_NO = config['PRB_NO']
# It previously identified it as str! casting to float did not work
SIGMA_NOISE = eval(config['SIGMA_NOISE'])
ETA_AREA = config['ETA_AREA']
X_LIM = config['X_LIM']
RU_PER_DU_NO = config['RU_PER_DU_NO']
DU_NO = config['DU_NO']
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
E = config['E']
T = config['T']
# DRL Hyperparameters
ALPHA_ACT = config['ALPHA_ACT']
BETA_ACT = config['BETA_ACT']
VAR = config['VAR']
DECAY_VAR = config['DECAY_VAR']
# ----------------------------------------
# %
# %% main class


class _main_:
    def __init__(self, E, T):
        SP = Specifications(USER_NO, SLICE_NO, CONST_D_MAX, CONST_R_MIN, PACKET_SIZE)
        self.mat_specs = SP._()
        # --------------------------------------
        self.loc_user_init = np.zeros([T, USER_NO, 2]) # initializing user_location... t=0 location will be changed randomly in the RadioCalc.user_location()
        # ---------
        RLOC_INIT = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY, X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY)
        self.mat_fh_links_capacity, self.mat_e2_links_capacity = RLOC_INIT.links_capacity()        
        # -----------------------------------------------
        self.mat_reward = np.zeros([E, T]) # -100 * np.ones([MC, T])
        self.mat_reward_user = np.zeros([E, USER_NO, T]) 
        self.mat_satisfied_prb_constraint = np.zeros([E, T])
        self.mat_satisfied_power_constraint = np.zeros([E, T])
        self.mat_satisfied_rate_constraint = np.zeros([E, T])
        self.mat_satisfied_delay_constraint = np.zeros([E, T])
        self.max_rate = np.zeros([E, T])
        self.max_inversed_delay = np.zeros([E, T]) 
        self.mat_ssl_u_rate = np.zeros([E, USER_NO, T])
        self.mat_ssl_u_delay = np.zeros([E, USER_NO, T])
        self.mat_ssl_user = np.zeros([E, USER_NO, T])
        self.mat_ssl_rate = np.zeros([E, T])
        self.mat_ssl_delay = np.zeros([E, T])
        self.mat_ssl = np.zeros([E, T])
        self.mat_episode_runtime = np.zeros([E, T])
        self.mat_chi = np.zeros([E, USER_NO, BS_NO, T]) # only get the latest MC

        self.shannon = np.zeros([E, USER_NO, T])

        self.mat_power = np.zeros([E, USER_NO, T])
        self.mat_gain = np.zeros([E, USER_NO, PRB_NO, T])
        self.mat_rho  = np.zeros([E, BS_NO, PRB_NO, USER_NO, T])
        self.mat_u_bs_dist = np.zeros([E, USER_NO, T])
        # -------------
        # Initialize matrices for SAC
        self.mat_used_prbs_per_user = np.zeros((E, USER_NO, T))
        self.mat_prb_util_per_bs = np.zeros((E, BS_NO, T))
        self.mat_b_connected_episodic = np.zeros((E, USER_NO, T))
        #####
        self.mat_chi_compressed = np.zeros((E, USER_NO, T))
        self.mat_rho_compressed = np.zeros((E, USER_NO, T))
        self.mat_p_compressed = np.zeros((E, USER_NO, T))

        # ---------------------------------------------------------
        self.monte_mat_delay_tot = np.zeros([E, USER_NO, T])
        # ----------obtaining the number of actions--------------
        self.num_actions = 3 * USER_NO # assuming that the user_association is conducted using a heuristic algorithm based on min_distance in user_location(self, t, loc_user)
        # ---------------------------------------------------------e
        self.s1 = BS_NO * USER_NO # H_b would be the avg of channel gains of the PRBs between u and b ### # USER_NO * BS_NO * PRB_NO # channel gain matrix (b,k,u) of t+1 # self.H = np.zeros([self.BS_NO, self.PRB_NO, self.USER_NO]) # defined in radio_calc.py -> user_location()
        self.s2 = self.s1 + 2 * USER_NO # vector of ssl for rate and delay per user in t-1
        self.s3 = self.s2 + 3 * USER_NO # vector of actions in t-1 (chi, rho, p)
        self.state_size = self.s3

    def _(self):
        #resetting the SAC agent here!            
        self.agent = Agent(ALPHA_ACT, BETA_ACT, self.num_actions, self.state_size)
        self.var = VAR # 
        self.decay_var = DECAY_VAR

        LC = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY, X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY)
        # -------------------------------------
        self.mat_bs_loc = LC.bs_location()
        self.mat_du_loc = LC.du_location()
        self.distances_ric_du = LC.ric_du_distance()
        self.distances_du_ru = LC.du_ru_distance()
        self.du_ru_adj_matrix, self.ric_du_adj_matrix = LC.adj_matrix()
        for e in range(E): #epsiodes (for RL to learn) 
            # agent should remain the same for all episodes # don't initialize the agent
            # first we need to initialize the locations/speeds, etc.
            for t in range(T-1):
                start_time = time.time()
                self.tt = t + 1
                self.reward = 0 # -100
                self.mat_reward_user[e, :, t] = 0
                # -----------------------------------------------                
                if t == 0:
                    self.loc_user = self.loc_user_init
                self.loc_user, self.H, self.mat_distance = LC.user_location(t, self.loc_user, self.mat_bs_loc)

                self.H_b = np.zeros([BS_NO, USER_NO])
                for u in range(USER_NO):
                    for b in range(BS_NO):
                        self.H_b[b, u] = np.average(self.H[b, :, u])
                
                if t==0:
                    SC = StateCalculation(self.H_b, np.zeros([USER_NO]), np.zeros([USER_NO]), np.zeros([USER_NO]), np.zeros([USER_NO]), np.zeros([USER_NO]))
                    self.state = SC._()
                else:   
                    # to add H(t+1) from LSTM prediction
                    SC = StateCalculation(self.H_b, self.mat_ssl_u_rate[e, :, t-1], self.mat_ssl_u_delay[e, :, t-1], self.mat_chi_compressed[e, :, t-1], self.mat_rho_compressed[e, :, t-1], self.mat_p_compressed[e, :, t-1])
                    self.state = SC._()
                self.mat_delay_tot = np.ones([USER_NO])
                self.mat_rate = np.zeros([USER_NO])
                # -----------------------------------------------------
                self.var = self.var * self.decay_var
                self.noise = np.random.randn(self.num_actions)
                self.noise = self.noise * self.var
                self.action = self.agent.choose_action(self.state)  # Choosing the action
                self.action += self.noise
                self.action = np.clip(self.action, -1, 1) # because of tanh activation function
                # -------Current state calculation---------------------
                MA = Mapping(self.action, self.mat_specs, self.H_b, USER_NO, BS_NO, PRB_NO, MAX_POWER)
                #chi
                self.chi_compressed, self.chi_num, self.chi = MA.user_association() # it is turned to a heuristic!
                self.mat_chi_compressed[e, :, t] = self.chi_compressed # between 0 and 1
                self.mat_chi[e, :, :, t] = self.chi # binary variable for b, u
                
                self.rho_compressed, self.rho_num, self.rho, self.unallocated_PRBs = MA.ran_prb_allocation() 
                self.mat_rho_compressed[e, :, t] = self.rho_compressed
                self.mat_rho[e,:,:,:,t] = self.rho

                for u in range(USER_NO):
                    self.mat_reward_user[e, u, t] -= self.unallocated_PRBs[u] / PRB_NO
                    #self.reward -= self.unallocated_PRBs[u] / PRB_NO

                total_unallocated_PRBs = np.sum(self.unallocated_PRBs)

                # Initialize PRB_utilization with zeros
                PRB_utilization = np.zeros([BS_NO])
                # Calculate PRB utilization for each base station
                for b in range(BS_NO):
                    PRB_utilization[b] = np.sum(self.rho[b, :, :]) / PRB_NO
                #maybe also give negative reward if avg utilization is low
                self.mat_prb_util_per_bs[e, :, t] = PRB_utilization

                for u in range(USER_NO):
                    for b in range(BS_NO):
                        if self.chi[u, b] == 1:
                            self.mat_used_prbs_per_user[e, :, t] = np.sum(self.rho[b, :, u])
                
                #p
                self.p_compressed, self.p_num, self.p = MA.ran_power_allocation()
                self.mat_p_compressed[e, :, t] = self.p_compressed # between 0 and 1

                for u in range(USER_NO):
                    self.mat_power[e, u, t] = np.sum(self.p[:, :, u]) # summing the power allocated to all user PRBs

                #########END OF ACTION ALLOCATION###############

                RC = RateCalculation(self.p, self.rho, self.H, self.chi, BS_NO, PRB_NO, USER_NO, SIGMA_NOISE, BW)
                self.mat_rate, self.mat_rate_prb, self.SINR_dB, self.signal_strength_dB, self.interference_dB, self.noise_plus_interference_dB, self.used_prbs_per_user_per_bs, self.num_prbs_used_per_user = RC._()
                self.shannon[e, :, t] = self.mat_rate  # b,k,u (in Mbps)

                D = Delay(self.mat_rate, FH_BW_CAPACITY, E2_BW_CAPACITY, self.mat_specs, self.chi, 
                        self.mat_distance, self.distances_ric_du, self.distances_du_ru, self.du_ru_adj_matrix, self.ric_du_adj_matrix, 
                        USER_NO, BS_NO, DU_NO)
                cnt_u, done_delay_all,  self.mat_delay_tot, is_fh_capacity_full, is_e2_capacity_full, flag_uu_failure_due_to_rate = D._()
                self.monte_mat_delay_tot[e, :, t] = self.mat_delay_tot 
                if np.sum(is_fh_capacity_full) > 0:
                    self.reward -= np.sum(is_fh_capacity_full) / (DU_NO * BS_NO)
                if np.sum(is_e2_capacity_full) > 0:
                    self.reward -= np.sum(is_e2_capacity_full) / DU_NO

                for u in range(USER_NO):
                    if flag_uu_failure_due_to_rate[u] == 1:
                        self.mat_reward_user[e, u, t] -= 1 # punish those users who had bad delay (although we fade the uu tx delay in measurements )
            
                self.sigma_SSL_R = 0
                cnt_rate_u = 0
                for s in range(SLICE_NO):
                    for u in range(USER_NO):
                        if self.mat_specs[u, 0] == s:
                            # min_rate specification
                            self.R_s = self.mat_specs[u, 1]
                            self.mat_ssl_u_rate[e, u, t] = (self.mat_rate[u] / self.R_s)
                            cnt_rate_u += (self.mat_rate[u] >= self.R_s)
                
                self.mat_satisfied_rate_constraint[e, t] = cnt_rate_u / USER_NO
                # Find the maximum value of mat_ssl_u_rate
                self.max_rate[e, t] = np.max(self.mat_ssl_u_rate[e, :, t])
                # Normalize mat_ssl_u_rate between 0 and 1
                self.mat_ssl_u_rate[e, :, t] = self.mat_ssl_u_rate[e, :, t] / (self.max_rate[e, t] if self.max_rate[e, t] != 0 else 1)# self.max_rate[e, t]
                ##########################
                self.sigma_SSL_D = 0
                cnt_delay_u = 0
                for s in range(SLICE_NO):
                    for u in range(USER_NO):
                        if self.mat_specs[u, 0] == s:
                            # max_tolerable_delay specification
                            self.D_s = self.mat_specs[u, 2]
                            self.mat_ssl_u_delay[e, u, t] = (self.D_s / self.mat_delay_tot[u])
                            cnt_delay_u += (self.mat_delay_tot[u] <= self.D_s)
                
                self.mat_satisfied_delay_constraint[e, t] = cnt_delay_u / USER_NO
                self.max_inversed_delay[e, t] = np.max(self.mat_ssl_u_delay[e, :, t])
                self.mat_ssl_u_delay[e, :, t] = self.mat_ssl_u_delay[e, :, t] / (self.max_inversed_delay[e, t] if self.max_inversed_delay[e, t] != 0 else 1) # normalized

                ### Now compute the SSL:
                self.mat_ssl_user[e, :, t] = (self.mat_ssl_u_rate[e, :, t] **(OMEGA_1)) * ((self.mat_ssl_u_delay[e, :, t] )**(1 - OMEGA_1)) # utility function

                self.mat_ssl_rate[e, t] = np.sum(self.mat_ssl_u_rate[e, :, t]) / (1 + np.sum(self.mat_ssl_u_rate[e, :, t]))
                self.mat_ssl_delay[e, t] = np.sum(self.mat_ssl_u_delay[e, :, t]) / (1 + np.sum(self.mat_ssl_u_delay[e, :, t]))

                self.mat_ssl[e, t] = np.average(self.mat_ssl_user[e, :, t]) # utility function

                self.reward += self.mat_ssl[e, t]

                self.mat_reward[e, t] = self.reward

                # ---------Next state calculation--------------
                LC_next = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY,
                              X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY)
                # self.loc_users_new, self.H_new, self.associator, self.mat_distance, self.mat_b_connected = LC_next.user_location(self.tt, self.loc_user, self.mat_bs_loc)
                self.loc_users_new, self.H_new, self.mat_distance = LC_next.user_location(self.tt, self.loc_user, self.mat_bs_loc)
                self.H_b_new = np.zeros([BS_NO, USER_NO])
                for u in range(USER_NO):
                    for b in range(BS_NO):
                        self.H_b_new[b, u] = np.average(self.H_new[b, :, u])
                # -----------------------------------------------------
                SC = StateCalculation(self.H_b_new, self.mat_ssl_u_rate[e, :, t], self.mat_ssl_u_delay[e, :, t], self.mat_chi_compressed[e, :, t], self.mat_rho_compressed[e, :, t], self.mat_p_compressed[e, :, t])
                self.next_state = SC._()
                self.next_state = self.next_state.astype('float16')
                # -------------------------------------
                self.agent.memorize(self.state, self.action,
                                    self.reward, self.next_state)
                self.agent.replay()
                # -------------------------------------------------------
                end_time = time.time()  # Record the end time of the loop
                # Storing the episode/timeslot runtime duration in seconds
                self.mat_episode_runtime[e,t] = end_time - start_time
            if e%100 == 0:
                LC.plot_user_movement(self.loc_user, self.mat_chi[e, :, :, :], T-1)
        return self.mat_rho, self.mat_u_bs_dist, self.shannon, self.mat_gain, self.mat_power, self.mat_reward, self.mat_satisfied_prb_constraint, self.mat_satisfied_power_constraint, self.mat_satisfied_delay_constraint, self.mat_satisfied_rate_constraint, self.mat_ssl_rate, self.mat_ssl_delay, self.mat_ssl, self.mat_episode_runtime, self.mat_rate, self.monte_mat_delay_tot, self.mat_used_prbs_per_user, self.mat_prb_util_per_bs, self.du_ru_adj_matrix, LC, self.mat_chi, self.loc_user, self.max_rate, self.max_inversed_delay, self.mat_ssl_u_rate, self.mat_ssl_u_delay


# %%
M = _main_(E, T)
mat_rho, mat_u_bs_dist, shannon, mat_gain, mat_power, mat_reward, mat_satisfied_prb_constraint, mat_satisfied_power_constraint, mat_satisfied_delay_constraint, mat_satisfied_rate_constraint, mat_ssl_rate, mat_ssl_delay, mat_ssl, mat_episode_runtime, mat_rate, monte_mat_delay_tot, mat_used_prbs_per_user, mat_prb_util_per_bs, du_ru_adj_matrix, LC, mat_chi, loc_user, max_rate, max_inversed_delay, mat_ssl_u_rate, mat_ssl_u_delay = M._()

# %%%%%%%

# save for later (use savez_compressed for compression)
filename = f'O-RAN SAC (Normal and Proactive), U={USER_NO}, PRB={PRB_NO}, BS={BS_NO}, E={E}, T={T}, VELOCITY={VELOCITY}, D_max={CONST_D_MAX}, R_min={CONST_R_MIN}.npz'
np.savez_compressed(filename, mat_rho=mat_rho, mat_u_bs_dist=mat_u_bs_dist, shannon=shannon, mat_gain=mat_gain, mat_power=mat_power, mat_reward=mat_reward, mat_satisfied_prb_constraint=mat_satisfied_prb_constraint, mat_satisfied_power_constraint=mat_satisfied_power_constraint,
                    mat_satisfied_delay_constraint=mat_satisfied_delay_constraint, mat_satisfied_rate_constraint=mat_satisfied_rate_constraint, mat_ssl_rate=mat_ssl_rate, mat_ssl_delay=mat_ssl_delay, mat_ssl=mat_ssl, mat_episode_runtime=mat_episode_runtime, mat_rate=mat_rate, monte_mat_delay_tot=monte_mat_delay_tot, mat_used_prbs_per_user=mat_used_prbs_per_user, mat_prb_util_per_bs=mat_prb_util_per_bs, du_ru_adj_matrix=du_ru_adj_matrix, mat_chi=mat_chi, max_rate=max_rate, max_inversed_delay=max_inversed_delay)
#%% %PLOTTING THE RESULTS%%
window_size = 50  # (for smoothing the curves in the plots)
#####
LC.visualize_ru_du_locations(du_ru_adj_matrix)
# %%%%RUNTIME DURATION%%%%%%%
plot_graph("Runtime Duration",
           [moving_average(1000*np.average(mat_episode_runtime, axis=1), window_size)],
           ['Average runtime duration: {:.2f} ms'.format(1000 * np.average(mat_episode_runtime))],
           ['blue'],
           ['solid'],
           "Episode",
           "Runtime duration (ms)")

# Calculate averages over Monte Carlo runs and users
avg_prbs_per_user = np.mean(mat_used_prbs_per_user, axis=(1,2))

# Plot the averages using your function
plot_graph('Avg PRBs used per user for SAC algorithm',
           [moving_average(avg_prbs_per_user , window_size)],
           ['SAC'],
           ['b'],
           ['-'],
           'T',
           'Average PRBs used per user')

# Fairness for PRBs
# Calculate PRB utilization for each user at each time step and for each MC run
prb_utilization = np.sum(mat_rho, axis=(2, 1))  # Sum over PRBs and BSs

# Calculate proportional fairness score for each time step
fairness_scores = np.zeros((E, T))

for e in range(E):
    for t in range(T):
        # Calculate Jain's fairness index for time step t in MC run mc
        sum_of_prbs = np.sum(prb_utilization[e, :, t])
        sum_of_squares = np.sum(prb_utilization[e, :, t] ** 2)
        # Handle potential division by zero or NaN
        if sum_of_prbs == 0 or np.isnan(sum_of_squares):
            fairness_scores[e, t] = 0  # Set fairness score to 0
        else:
            fairness_scores[e, t] = (sum_of_prbs ** 2) / (BS_NO * sum_of_squares)

avg_fairness_scores = np.mean(fairness_scores, axis=1)
normalized_fairness_scores = (avg_fairness_scores - np.min(avg_fairness_scores)) / (np.max(avg_fairness_scores) - np.min(avg_fairness_scores))


plot_graph("Jain's fairness index for PRB allocation",
           [moving_average(normalized_fairness_scores , window_size)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Episode",
           "Mean fairness score")

# %%REWARD%%%%

plot_graph("Mean episodic rewards",
           [moving_average(np.average(mat_reward, axis=1), window_size)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Episode",
           "Mean episodic rewards")
# plot_graph("Mean episodic rewards",
#            [mean_ep_rewardall, mean_ep_rewardall_pred],
#            ['SAC', 'Proactive SAC'],
#            ['blue', 'green'],
#            ['solid', 'dotted'],
#            "Episode",
#            "Mean episodic rewards")
# %%CONSTRAINT SATISFACTION%
plot_graph("Constraint Satisfaction",
           [moving_average(np.average(mat_satisfied_prb_constraint, axis=1), window_size), 
            moving_average(np.average(mat_satisfied_power_constraint, axis=1), window_size),
            moving_average(np.average(mat_satisfied_delay_constraint, axis=1), window_size),
            moving_average(np.average(mat_satisfied_rate_constraint, axis=1), window_size)],
           ['PRB (SAC)',
            'Power (SAC)',
            'Delay (SAC)',
            'Rate (SAC)'],
           ['red', 'blue', 'green', 'orange'],
           ['solid', 'solid', 'solid', 'solid'],
           "Episode",
           "Constraint Satisfaction Rate")

plot_graph("SSL Metrics",
           [moving_average(np.average(mat_ssl_rate, axis=1), window_size),
            moving_average(np.average(mat_ssl_delay, axis=1), window_size),
            moving_average(np.average(mat_ssl, axis=1), window_size)],
           ['Rate (SAC)',
            'Delay (SAC)',
            'SSL (SAC)'],
           ['blue', 'green', 'orange'],
           ['solid', 'solid', 'solid'],
           "Episode",
           "SSL Metrics")
# %%%%%Delay%%%%%%
# Calculate mean delay over users for SAC
mean_delay_sac = np.mean(np.mean(monte_mat_delay_tot, axis=1), axis=1)

plot_graph("Comparison of Average E2E Delay (SAC)",
           [moving_average(mean_delay_sac , window_size)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Timestep",
           "Average E2E Delay (ms)")

mean_rate_sac = np.mean(np.mean(shannon, axis=1), axis=1)
plot_graph("Average Data Rate (SAC)",
           [moving_average(mean_rate_sac , window_size)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Timestep",
           "Average Data Rate (Mbps)")

#####################
mean_power_sac = np.mean(np.mean(mat_power, axis=1), axis=1)
plot_graph("Average of total allocated power to each user (SAC)",
           [moving_average(mean_power_sac , window_size)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Timestep",
           "Average of total allocated power to each user (W)")

###
average_rate = np.mean(shannon, axis=2)

for user_idx in range(USER_NO):
    plt.plot(moving_average(range(E), window_size), moving_average(average_rate[:,user_idx], window_size), label=f'User {user_idx+1}')
plt.xlabel('E')
plt.ylabel('Rate (Mbps)')
plt.title('Average rate for individual users')
plt.legend()
plt.show()

###
average_rate_ssl_u = np.mean(mat_ssl_u_rate, axis=2)

for user_idx in range(USER_NO):
    plt.plot(moving_average(range(E), window_size), moving_average(average_rate_ssl_u[:,user_idx], window_size), label=f'User {user_idx+1}')
plt.xlabel('E')
plt.ylabel('SSL_rate_u')
plt.title('Normalized rate satisfaction')
plt.legend()
plt.show()

###
average_delay_ssl_u = np.mean(mat_ssl_u_delay, axis=2)

for user_idx in range(USER_NO):
    plt.plot(moving_average(range(E), window_size), moving_average(average_delay_ssl_u[:,user_idx], window_size), label=f'User {user_idx+1}')
plt.xlabel('E')
plt.ylabel('SSL_delay_u')
plt.title('Normalized delay satisfaction')
plt.legend()
plt.show()


print(style.UNDERLINE + "Total time for {} timesteps ({} users) in {} Episdoes (aka iterations or epochs): {}".format(T, USER_NO, E, convert_seconds(np.sum(mat_episode_runtime))))

