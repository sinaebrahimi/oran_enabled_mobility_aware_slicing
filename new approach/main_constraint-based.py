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
from initialization import Specifications
from radio_calc import Location, RateCalculation
from e2e_calc import Mapping, Delay, StateCalculation
from sac_torch import Agent
np.random.seed(1371) # some random number
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# or "1"; change the GPU for multiple simulations (We have 0 and 1 in K80 (zeus401 and zeus402))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ------Loading the parameters-----------
# Define the path to the configuration file
# config_file = 'sac-lstm/config_lstm.yaml' #
config_file = 'new approach/config.yaml'

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
# PACKET_NO = config['PACKET_NO']
RAYLEIGH_SCALE = config['RAYLEIGH_SCALE']
E = config['E']
T = config['T']
#MC = config['MC']
# DRL Hyperparameters
XI = config['XI']
PSI = config['PSI']
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
        #self.mat_loc_user = np.zeros([MC, T, USER_NO, 2])
        # ---------
        RLOC_INIT = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY, X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY)
        self.mat_fh_links_capacity, self.mat_e2_links_capacity = RLOC_INIT.links_capacity()        
        # -----------------------------------------------
        self.chi = np.zeros([USER_NO, BS_NO]) # user-bs associator
        self.chi_num = np.zeros([USER_NO])
        self.p = np.zeros([BS_NO, PRB_NO, USER_NO])
        self.rho = np.zeros([BS_NO, PRB_NO, USER_NO])
        # self.mat_reward = np.zeros([T])
        self.mat_reward = np.zeros([E, T]) # -100 * np.ones([MC, T])
        self.mat_reward_user = np.zeros([E, USER_NO, T]) 
        self.mat_satisfied_prb_constraint = np.zeros([E, T])
        self.mat_satisfied_power_constraint = np.zeros([E, T])
        self.mat_satisfied_rate_constraint = np.zeros([E, T])
        self.mat_satisfied_delay_constraint = np.zeros([E, T])
        # self.mat_satisfied_fh_link_capacity_constraint = np.zeros([MC, T])
        # self.mat_satisfied_e2_link_capacity_constraint = np.zeros([MC, T])
        self.max_rate = np.zeros([E, T])
        self.max_inversed_delay = np.zeros([E, T]) 
        self.mat_ssl_u_rate = np.zeros([E, USER_NO, T])
        self.mat_ssl_u_delay = np.zeros([E, USER_NO, T])        
        self.mat_ssl_u_total = np.zeros([E, USER_NO, T])      
        self.mat_fittingness_u_rate = np.zeros([E, USER_NO, T])
        self.mat_fittingness_u_delay = np.zeros([E, USER_NO, T])
        self.mat_ssl_user = np.zeros([E, USER_NO, T])
        self.mat_ssl_rate = np.ones([E, T])
        self.mat_ssl_delay = np.ones([E, T])
        self.mat_ssl = np.ones([E, T])
        self.mat_episode_runtime = np.zeros([E, T])
        self.mat_chi = np.zeros([E, USER_NO, BS_NO, T]) # only get the latest MC
        self.H_b = np.zeros([BS_NO, USER_NO])

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
        self.mat_count_handovers = np.zeros((E, USER_NO))
        # ---------------------------------------------------------
        self.monte_mat_delay_tot = np.zeros([E, USER_NO, T])
        # ----------obtaining the number of actions--------------
        self.num_actions = 3 * USER_NO # assuming that the user_association is conducted using a heuristic algorithm based on min_distance in user_location(self, t, loc_user)
        # ---------------------------------------------------------e
        self.s1 = BS_NO * USER_NO # H_b would be the avg of channel gains of the PRBs between u and b ### # USER_NO * BS_NO * PRB_NO # channel gain matrix (b,k,u) of t+1 # self.H = np.zeros([self.BS_NO, self.PRB_NO, self.USER_NO]) # defined in radio_calc.py -> user_location()
        self.s2 = self.s1 + USER_NO # vector of ssl for rate and delay per user in t-1
        self.s3 = self.s2 + 3 * USER_NO # vector of actions in t-1 (chi, rho, p)
        self.state_size = self.s3

    
    def heuristic_initial_allocation(self):
        # Initial user association based on channel gain      

        for u in range(USER_NO):
            best_bs = np.argmax(self.H_b[:, u])
            self.chi[u, best_bs] = 1
        self.chi_num = np.argmax(self.chi, axis=1)

        # Initial PRB allocation (simple round-robin)
        for b in range(BS_NO):
            users_connected_to_b = np.where(self.chi_num == b)[0]
            if len(users_connected_to_b) > 0:
                prbs_per_user = PRB_NO // len(users_connected_to_b)
                for i, u in enumerate(users_connected_to_b):
                    self.rho[b, i * prbs_per_user:(i + 1) * prbs_per_user, u] = 1


        # Initial power allocation (equal power allocation)
        for b in range(BS_NO):
            users_connected_to_b = np.where(self.chi_num == b)[0]
            if len(users_connected_to_b) > 0:
                power_per_user = MAX_POWER / len(users_connected_to_b)
                for u in users_connected_to_b:
                    for k in range(PRB_NO):
                        if self.rho[b, k, u] == 1:
                            self.p[b, k, u] = power_per_user / PRB_NO

        return self.chi, self.rho, self.p
    
    def adjust_allocations(self, actions):
        # Actions are changes to the initial allocations
        delta_chi = actions[:USER_NO]
        delta_rho = actions[USER_NO:USER_NO * 2]
        delta_p = actions[USER_NO * 2:USER_NO * 3]

        # Adjust user association
        self.chi_num += delta_chi
        self.chi_num = np.clip(self.chi_num, 0, BS_NO - 1)  # Ensure within bounds
        self.chi = np.zeros([USER_NO, BS_NO])
        self.chi[np.arange(USER_NO), self.chi_num.astype(int)] = 1

        # Adjust PRB allocation
        for u in range(USER_NO):
            b = int(self.chi_num[u])
            self.rho[b, :, u] += delta_rho[u]
            self.rho[b, :, u] = np.clip(self.rho[b, :, u], 0, 1)  # Ensure within bounds

        # Adjust power allocation
        for u in range(USER_NO):
            b = int(self.chi_num[u])
            self.p[b, :, u] += delta_p[u]
            self.p[b, :, u] = np.clip(self.p[b, :, u], 0, MAX_POWER / PRB_NO)  # Ensure within bounds

        return self.chi, self.rho, self.p

    def compute_ssl(self, chi, rho, p):
        # Calculate rate
        RC = RateCalculation(p, rho, self.H, chi, BS_NO, PRB_NO, USER_NO, SIGMA_NOISE, BW)
        mat_rate, mat_rate_prb, SINR_dB, signal_strength_dB, interference_dB, noise_plus_interference_dB, used_prbs_per_user_per_bs, num_prbs_used_per_user = RC._()
        self.shannon[self.current_episode, :, self.current_timestep] = mat_rate  # b,k,u (in Mbps)

        # Calculate delay
        D = Delay(mat_rate, FH_BW_CAPACITY, E2_BW_CAPACITY, self.mat_specs, chi, self.mat_distance, self.distances_ric_du, self.distances_du_ru, self.du_ru_adj_matrix, self.ric_du_adj_matrix, USER_NO, BS_NO, DU_NO)
        cnt_u, done_delay_all, mat_delay_tot, is_fh_capacity_full, is_e2_capacity_full, flag_uu_failure_due_to_rate = D._()
        self.monte_mat_delay_tot[self.current_episode, :, self.current_timestep] = mat_delay_tot

        # Initialize SSL metrics
        cnt_rate_passed_u = 0
        cnt_delay_passed_u = 0

        for s in range(SLICE_NO):
            for u in range(USER_NO):
                if self.mat_specs[u, 0] == s:
                    # Rate satisfaction
                    R_s = self.mat_specs[u, 1]
                    self.mat_ssl_u_rate[u] = (mat_rate[u] / R_s)
                    temp_rate_satisfaction_ratio = (mat_rate[u] / R_s) ** XI
                    self.mat_fittingness_u_rate[self.current_episode, u, self.current_timestep] = temp_rate_satisfaction_ratio / (1 + temp_rate_satisfaction_ratio)
                    cnt_rate_passed_u += (mat_rate[u] >= R_s)
                    
                    # Delay satisfaction
                    D_s = self.mat_specs[u, 2]
                    self.mat_ssl_u_delay[u] = (D_s / mat_delay_tot[u])
                    temp_delay_satisfaction_ratio = (D_s / mat_delay_tot[u]) ** XI
                    self.mat_fittingness_u_delay[c] = temp_delay_satisfaction_ratio / (1 + temp_delay_satisfaction_ratio)
                    cnt_delay_passed_u += (mat_delay_tot[u] <= D_s)
        
        self.mat_satisfied_rate_constraint[self.current_episode, self.current_timestep] = cnt_rate_passed_u / USER_NO
        self.mat_satisfied_delay_constraint[self.current_episode, self.current_timestep] = cnt_delay_passed_u / USER_NO
        
        # Calculate total SSL for each user
        for u in range(USER_NO):
            self.mat_ssl_u_total[self.current_episode, u, self.current_timestep] = (self.mat_fittingness_u_rate[self.current_episode, u, self.current_timestep] ** OMEGA_1) * (self.mat_fittingness_u_delay[self.current_episode, u, self.current_timestep] ** (1 - OMEGA_1))
        
        # Aggregate SSL across all users
        ssl_rate = np.prod([self.mat_fittingness_u_rate[self.current_episode, u, self.current_timestep] for u in range(USER_NO)]) ** (1 / USER_NO)
        ssl_delay = np.prod([self.mat_fittingness_u_delay[self.current_episode, u, self.current_timestep] for u in range(USER_NO)]) ** (1 / USER_NO)
        ssl = np.prod([self.mat_ssl_u_total[self.current_episode, u, self.current_timestep]  for u in range(USER_NO)]) ** (1 / USER_NO)

        return ssl, ssl_rate, ssl_delay


    def accept_if_improved(self, new_ssl, prev_ssl, chi, rho, p):
        if new_ssl > prev_ssl:
            self.chi, self.rho, self.p = chi, rho, p
            return True
        return False
    
    def update_allocations(self, actions, prev_ssl):
        # Adjust allocations based on actions
        chi, rho, p = self.adjust_allocations(actions)

        # Compute new SSL
        new_ssl, new_ssl_rate, new_ssl_delay = self.compute_ssl(chi, rho, p)

        # Accept new allocations if SSL improves
        improved = self.accept_if_improved(new_ssl, prev_ssl, chi, rho, p)

        return improved, new_ssl

    def _(self):
        #resetting the SAC agent here!  
        print(E,T)          
        self.agent = Agent(ALPHA_ACT, BETA_ACT, self.num_actions, self.state_size)

        LC = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY, X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY)
        # -------------------------------------
        self.mat_bs_loc = LC.bs_location()
        self.mat_du_loc = LC.du_location()
        self.distances_ric_du = LC.ric_du_distance()
        self.distances_du_ru = LC.du_ru_distance()
        self.du_ru_adj_matrix, self.ric_du_adj_matrix = LC.adj_matrix()
        for e in range(E):
            self.current_episode = e
            #maybe reset the LSTM as well
            count_handovers = 0
            prev_ssl = 0

            #loc_user_m = np.zeros([T, USER_NO, 2])        

            #FOR episodes! for RL
            # agent should remain the same for all episodes # don't initialize the agent
            # first we need to initialize the locations/speeds, etc.
            for t in range(T-1):
                self.current_timestep = t

                start_time = time.time()
                self.tt = t + 1
                # starting with a negative award, aiming to learn more in initial episodes; Not sure if it is necessary (due to line 495)
                self.reward = 0 # -100
                self.mat_reward_user[e, :, t] = 0

                
                # -----------------------------------------------                
                if t == 0:
                    self.loc_user = self.loc_user_init
                    self.loc_user, self.H, self.mat_distance = LC.user_location(t, self.loc_user, self.mat_bs_loc)
                    
                    for u in range(USER_NO):
                        for b in range(BS_NO):
                            self.H_b[b, u] = np.average(self.H[b, :, u])

                    SC = StateCalculation(self.H_b, np.zeros([USER_NO]), np.zeros([USER_NO]), np.zeros([USER_NO]), np.zeros([USER_NO]))
                    self.state = SC._()

                    chi, rho, p = self.heuristic_initial_allocation()
                    new_ssl, new_ssl_rate, new_ssl_delay = self.compute_ssl(chi, rho, p)
                    prev_ssl = new_ssl


                else:
                    self.loc_user, self.H, self.mat_distance = LC.user_location(t, self.loc_user, self.mat_bs_loc)
                    for u in range(USER_NO):
                        for b in range(BS_NO):
                            self.H_b[b, u] = np.average(self.H[b, :, u])
                    SC = StateCalculation(self.H_b, self.mat_ssl_u_total[e, :, t-1], self.mat_chi_compressed[e, :, t-1], self.mat_rho_compressed[e, :, t-1], self.mat_p_compressed[e, :, t-1])
                    self.state = SC._()
                    actions = self.agent.choose_action(self.state)
                    improved, new_ssl = self.update_allocations(actions, prev_ssl)
                    if improved:
                        prev_ssl = new_ssl
                    else:
                        chi, rho, p = self.chi, self.rho, self.p  # revert back to previous time step's allocations

                
                    mat_ssl[e, t] = prev_ssl
                    self.reward = 100 * self.mat_ssl[e, t] 
                    self.mat_reward[e, t] = self.reward

                    # ---------Next state calculation--------------
                    LC_next = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY,
                                X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY)
                    self.loc_users_new, self.H_new, self.mat_distance = LC_next.user_location(self.tt, self.loc_user, self.mat_bs_loc)
                    self.H_b_new = np.zeros([BS_NO, USER_NO])
                    for u in range(USER_NO):
                        for b in range(BS_NO):
                            self.H_b_new[b, u] = np.average(self.H_new[b, :, u])
                    # -----------------------------------------------------
                    SC = StateCalculation(self.H_b_new, self.mat_ssl_u_total[e, :, t], self.mat_chi_compressed[e, :, t], self.mat_rho_compressed[e, :, t], self.mat_p_compressed[e, :, t])
                    self.next_state = SC._()
                    self.next_state = self.next_state.astype('float16')
                    # -------------------------------------
                    self.agent.memorize(self.state, actions,
                                        self.reward, self.next_state)
                    self.agent.replay()

                # -------------------------------------------------------
                end_time = time.time()  # Record the end time of the loop
                # Storing the episode/timeslot runtime duration in seconds
                self.mat_episode_runtime[e,t] = end_time - start_time
                
            if e%100 == 0:
                print(style.CYAN + ': Episode {}, reward= {}'.format(e, self.reward))
                #print(style.CYAN + 'Total Handovers  over all timesteps: Episode {}= {} HOs, reward= {}'.format(e, count_handovers, self.reward))

                # LC.plot_user_movement(self.loc_user, self.mat_chi[e, :, :, :], T-1)
        return self.mat_rho, self.mat_u_bs_dist, self.shannon, self.mat_gain, self.mat_power, self.mat_reward, self.mat_satisfied_prb_constraint, self.mat_satisfied_power_constraint, self.mat_satisfied_delay_constraint, self.mat_satisfied_rate_constraint, self.mat_ssl_rate, self.mat_ssl_delay, self.mat_ssl, self.mat_episode_runtime, self.mat_rate, self.monte_mat_delay_tot, self.mat_used_prbs_per_user, self.mat_prb_util_per_bs, self.du_ru_adj_matrix, LC, self.mat_chi, self.loc_user, self.max_rate, self.max_inversed_delay, self.mat_ssl_u_rate, self.mat_ssl_u_delay, self.mat_fittingness_u_rate, self.mat_fittingness_u_delay, self.mat_specs, self.mat_count_handovers, self.mat_ssl_u_total


# %%
M = _main_(E, T)
mat_rho, mat_u_bs_dist, shannon, mat_gain, mat_power, mat_reward, mat_satisfied_prb_constraint, mat_satisfied_power_constraint, mat_satisfied_delay_constraint, mat_satisfied_rate_constraint, mat_ssl_rate, mat_ssl_delay, mat_ssl, mat_episode_runtime, mat_rate, monte_mat_delay_tot, mat_used_prbs_per_user, mat_prb_util_per_bs, du_ru_adj_matrix, LC, mat_chi, loc_user, max_rate, max_inversed_delay, mat_ssl_u_rate, mat_ssl_u_delay, mat_fittingness_u_rate, mat_fittingness_u_delay, mat_specs, mat_count_handovers, mat_ssl_u_total = M._()
# mat_rho, mat_u_bs_dist, mat_u_bs_dist_pred, shannon, shannon_pred, mat_gain, mat_gain_pred, mat_power, mat_power_pred, mat_reward, mat_reward_pred, mat_satisfied_prb_constraint, mat_satisfied_prb_constraint_pred, mat_satisfied_power_constraint, mat_satisfied_power_constraint_pred, mat_satisfied_delay_constraint, mat_satisfied_delay_constraint_pred, mat_ssl_rate, mat_ssl_rate_pred, mat_ssl_delay, mat_ssl_delay_pred, mat_ssl, mat_ssl_pred, mat_episode_runtime, mat_rate, mat_rate_pred, monte_mat_delay_tot, monte_mat_delay_tot_pred, mat_used_prbs_per_user, mat_used_prbs_per_user_per_bs, mat_used_prbs_per_user_pred, mat_used_prbs_per_user_per_bs_pred, du_ru_adj_matrix, LC, mat_associator, loc_user = M._()

# %%%%%%%

# save for later (use savez_compressed for compression)
filename = f'O-RAN SAC, U={USER_NO}, PRB={PRB_NO}, BS={BS_NO}, E={E}, T={T}, VELOCITY={VELOCITY}, D_max={CONST_D_MAX}, R_min={CONST_R_MIN}.npz'
np.savez_compressed(filename, mat_rho=mat_rho, mat_u_bs_dist=mat_u_bs_dist, shannon=shannon, mat_gain=mat_gain, mat_power=mat_power, mat_reward=mat_reward, mat_satisfied_prb_constraint=mat_satisfied_prb_constraint, mat_satisfied_power_constraint=mat_satisfied_power_constraint,
                    mat_satisfied_delay_constraint=mat_satisfied_delay_constraint, mat_satisfied_rate_constraint=mat_satisfied_rate_constraint, mat_ssl_rate=mat_ssl_rate, mat_ssl_delay=mat_ssl_delay, mat_ssl=mat_ssl, mat_episode_runtime=mat_episode_runtime, mat_rate=mat_rate, monte_mat_delay_tot=monte_mat_delay_tot, mat_used_prbs_per_user=mat_used_prbs_per_user, mat_prb_util_per_bs=mat_prb_util_per_bs, du_ru_adj_matrix=du_ru_adj_matrix, mat_chi=mat_chi, max_rate=max_rate, max_inversed_delay=max_inversed_delay, mat_fittingness_u_rate=mat_fittingness_u_rate, mat_fittingness_u_delay=mat_fittingness_u_delay, mat_specs=mat_specs, mat_count_handovers=mat_count_handovers, mat_ssl_u_total=mat_ssl_u_total)

window_size = 100
plot_graph("Mean episodic rewards",
           [moving_average(np.average(mat_reward, axis=1), window_size)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Episode",
           "Mean episodic rewards")



print(style.UNDERLINE + "Total time for {} timesteps ({} users) in {} Episdoes (aka iterations or epochs): {}".format(T, USER_NO, E, convert_seconds(np.sum(mat_episode_runtime))))

