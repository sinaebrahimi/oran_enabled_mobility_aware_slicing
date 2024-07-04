# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import yaml
# -----import from other files-----------
from style import style, convert_seconds # for printing styles (e.g., bold, different colors, etc.)
from plot_assistant import plot_graph, moving_average
from initialization import Specifications
from radio_calc import Location, RateCalculation
from e2e_calc import Mapping, Delay, StateCalculation
from sac_torch import Agent
np.random.seed(1371) # some random number
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # or "1"; change the GPU for multiple simulations (We have 0 and 1 in K80 (zeus401 and zeus402))
# ------Loading the parameters-----------
config_file = 'config.yaml' # Define the path to the configuration file
# config_file = '0new_mdp_design/config.yaml' # Define the path to the configuration file

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
RU_PER_DU_NO = config['RU_PER_DU_NO'] #BS_NO = config['BS_NO']
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
PACKET_SIZE = config['PACKET_SIZE'] # PACKET_NO = config['PACKET_NO']
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
IOTA = config['IOTA']

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
        self.mat_no_of_satisfied_delay_and_rate_constraint = np.zeros([E, T])
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
        self.log_geometric_mean = np.zeros([E, T])        
        self.regularization_term = np.zeros([E, T])   
        self.logarithmic_reward = np.zeros([E, T])
        self.user_assignment_penalty_per_user = np.zeros([E, USER_NO, T])
        self.total_user_assignment_penalty = np.zeros([E, T])

        ####
        self.mat_episode_runtime = np.zeros([E, T])
        self.mat_chi = np.zeros([E, USER_NO, BS_NO, T]) # only get the latest MC
        self.mat_chi_action = np.zeros([E, USER_NO, T]) # self.mat_chi_action, self.mat_p_action, self.mat_rho_action
        self.shannon = np.zeros([E, USER_NO, T])
        self.mat_sum_power = np.zeros([E, USER_NO, T])
        self.mat_p  = np.zeros([E, BS_NO, PRB_NO, USER_NO, T])
        self.mat_p_action = np.zeros([E, USER_NO, PRB_NO, T])
        self.mat_gain = np.zeros([E, BS_NO, USER_NO, T])
        self.mat_rho  = np.zeros([E, BS_NO, PRB_NO, USER_NO, T])
        self.mat_rho_action = np.zeros([E, USER_NO, PRB_NO, T])
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
        self.monte_mat_delay_tot = np.zeros([E, USER_NO, T])
        self.angle_historic = np.zeros([E, USER_NO, T])
        self.prb_util_t = np.zeros([E, BS_NO, T])
        self.power_util_t = np.zeros([E, BS_NO, T])
        self.prb_ratio_t = np.zeros([E, USER_NO, T])
        self.power_ratio_t = np.zeros([E, USER_NO, T])
        # # ----------obtaining the number of actions--------------
        # # self.e1 = USER_NO * BS_NO # user_association()
        # # self.e2 = self.e1 + BS_NO * PRB_NO * USER_NO # ran_prb_allocation()
        # # self.e3 = self.e2 + BS_NO * PRB_NO * USER_NO # ran_power_allocation()
        # self.num_actions = 3 * USER_NO # assuming that the user_association is conducted using a heuristic algorithm based on min_distance in user_location(self, t, loc_user)
        self.e1 = USER_NO # user_association()
        self.e2 = self.e1 + USER_NO * PRB_NO # ran_prb_allocation()
        self.e3 = self.e2 + USER_NO * PRB_NO # ran_power_allocation()
        self.num_actions = self.e3 # assuming that the user_association is conducted using a heuristic algorithm based on min_distance in user_location(self, t, loc_user)
        # ---------------------------------------------------------e
        # self.s0 = BS_NO * USER_NO # H_b would be the avg of channel gains of the PRBs between u and b ### # USER_NO * BS_NO * PRB_NO # channel gain matrix (b,k,u) of t+1 # self.H = np.zeros([self.BS_NO, self.PRB_NO, self.USER_NO]) # defined in radio_calc.py -> user_location()
        self.s1 = USER_NO # vector of mat_ssl_u_total in t-1
        self.s2 = self.s1 + BS_NO # vector of prb_util per BS in t-1
        self.s3 = self.s2 + USER_NO # vector of prb_ratio per user in t-1
        self.s4 = self.s3 + BS_NO # vector of power_util per BS in t-1v 
        self.s5 = self.s4 + USER_NO # ector of power_ratio per user in t-1
        # self.s3 = self.s2 + self.num_actions
        self.state_size = self.s5

    def calculate_penalty(self): # H_b, USER_NO, BS_NO, chi_num
        penalty = 0
        penalty_per_user = np.zeros([USER_NO])

        ##Normalized-based penalization###
        # for u in range(USER_NO):
        #     highest_gain_value = np.max(self.H_b[:, u])
        #     selected_gain_value = self.H_b[int(self.chi_num[u]), u]
        #     lowest_gain_value = np.min(self.H_b[:, u])

        #     if selected_gain_value != highest_gain_value:
        #         penalty_per_user[u] = (highest_gain_value - selected_gain_value) / (highest_gain_value - lowest_gain_value)
        #         penalty += (1 / USER_NO) * penalty_per_user[u]

        ##Rank-based penalization###
        for u in range(USER_NO):
            user_channel_gains = self.H_b_normalized[:, u]
            sorted_indices = np.argsort(-user_channel_gains)  # Sort in descending order
            selected_index = int(self.chi_num[u])
            rank = np.where(sorted_indices == selected_index)[0][0]
            normalized_rank = rank / (BS_NO - 1)
            penalty_per_user[u] = normalized_rank
            penalty += (1 / USER_NO) * normalized_rank
        return penalty, penalty_per_user

    def _(self):
        #resetting the SAC agent here!  
        print(E,T)
        # self.var = VAR # .9995 #experiment .9995 and .995 # can determine the ratio of exploration to exploitation
        # self.decay_var = DECAY_VAR
        LC = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY, X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY)
        # -------------------------------------
        self.mat_bs_loc = LC.bs_location()
        self.mat_du_loc = LC.du_location()
        self.distances_ric_du = LC.ric_du_distance()
        self.distances_du_ru = LC.du_ru_distance()
        self.du_ru_adj_matrix, self.ric_du_adj_matrix = LC.adj_matrix()
        for e in range(E):
            # self.agent = Agent(ALPHA_ACT, BETA_ACT, self.num_actions, self.state_size)
            #maybe reset the LSTM as well
            self.agent = Agent(ALPHA_ACT, BETA_ACT, self.num_actions, self.state_size)
            count_handovers = 0
            action_memory = np.zeros([T, self.num_actions])

            #FOR episodes! for RL
            # agent should remain the same for all episodes # don't initialize the agent
            # first we need to initialize the locations/speeds, etc.
            for t in range(T-1):
                start_time = time.time()
                self.tt = t + 1
                # starting with a negative award, aiming to learn more in initial episodes; Not sure if it is necessary (due to line 495)
                self.reward = 0 # -1 # 0 # -1 # -100
                self.mat_delay_tot = np.ones([USER_NO])
                self.mat_rate = np.zeros([USER_NO])
                # ----------------------------------------------- 
                # User location generation     and states 
                # 
                self.H_b = np.zeros([BS_NO, USER_NO])
                self.H_b_normalized = np.zeros([BS_NO, USER_NO])         

                if t==0:
                    self.loc_user = self.loc_user_init
                    self.loc_user, self.H, self.mat_distance, self.angle = LC.user_location(t, self.loc_user, self.mat_bs_loc, np.zeros(USER_NO))
                    self.angle_historic[e,:,t] = self.angle                    

                    SC = StateCalculation(0.5 * np.ones([USER_NO]), 0.5 * np.ones([BS_NO]), 0.5 * np.ones([USER_NO]), 
                                          0.5 * np.ones([BS_NO]), 0.5 * np.ones([USER_NO]))
                    self.state = SC._()
                else: 
                    self.loc_user = self.loc_users_new # get it from the calculated location for s' (end of prev time step)
                    self.H = self.H_new
                    self.mat_distance = self.mat_distance_new

                    # SC = StateCalculation(self.H_b_normalized, self.mat_ssl_u_total[e, :, t-1], 
                    #                       self.mat_chi[e, :, :, t-1], self.mat_rho[e,:,:,:,t-1], self.mat_p[e, :, :, :, t-1] )
                    SC = StateCalculation(self.mat_ssl_u_total[e, :, t-1], self.prb_util_t[e, :, t-1], self.prb_ratio_t[e, :, t-1], 
                                          self.power_util_t[e, :, t-1], self.power_ratio_t[e, :, t-1])
                    self.state = SC._()

                # Calculate H_b from the states
                for u in range(USER_NO):
                    for b in range(BS_NO):
                        self.H_b[b, u] = np.average(self.H[b, :, u])
                
                min_val = np.min(self.H_b)
                max_val = np.max(self.H_b)

                # Perform min-max normalization
                self.H_b_normalized = (self.H_b - min_val) / (max_val - min_val)
                self.mat_gain[e, :, :, t] = self.H_b
                ###########ACTIONS  ######
                if t==0:
                    #self.action = self.agent.choose_action(self.state)
                    self.action = np.zeros([self.num_actions])
                    MA = Mapping(self.action, self.mat_specs, self.H_b_normalized, USER_NO, BS_NO, PRB_NO, MAX_POWER)
                    self.chi_num, self.chi, self.chi_action = MA.user_association_t0()
                    self.rho_num, self.rho, self.rho_action = MA.ran_prb_allocation_t0() 
                    self.p_num, self.p, self.p_action = MA.ran_power_allocation_t0()

                    # generate the self.action from chi_action + rho_action + p_action
                    # to use self.action at the end of time step to store it in self.agent.memorize 
                    # Flatten the actions
                    chi_action_flat = self.chi_action.flatten()
                    rho_action_flat = self.rho_action.flatten()
                    p_action_flat = self.p_action.flatten()

                    # Assign the flattened actions to the respective slices of self.action
                    self.action[:self.e1] = chi_action_flat
                    self.action[self.e1:self.e2] = rho_action_flat
                    self.action[self.e2:self.e3] = p_action_flat                    
                    
                else:
                    if False: # to use when we want to duplicate actions
                    #if self.mat_satisfied_delay_constraint[e, t-1] == 1.0 and self.mat_satisfied_rate_constraint[e, t-1] == 1.0:
                        #repeat
                        print(style.MAGENTA + 'DUPLICATED! Episode: {}, Timestep: {}, Reward: {}'.format(e, t, self.mat_reward[e, t-1]))
                        # self.action = action_memory[t-1, :]

                        # self.chi = self.mat_chi[e, :, :, t-1]
                        # self.chi_num = self.mat_b_connected_episodic[e, :, t-1]

                        # self.rho = self.mat_rho[e,:,:,:,t-1]
                        # self.p = self.mat_p[e, :, :, :, t-1] 
                    else:
                        self.action = self.agent.choose_action(self.state)
                        MA = Mapping(self.action, self.mat_specs, self.H_b_normalized, USER_NO, BS_NO, PRB_NO, MAX_POWER)
                        self.chi_num, self.chi, self.chi_action = MA.user_association(self.mat_b_connected_episodic[e, :, t-1])
                        self.rho_num, self.rho, self.rho_action = MA.ran_prb_allocation() 
                        self.p_num, self.p, self.p_action = MA.ran_power_allocation()
                #######################


                self.mat_chi[e, :, :, t] = self.chi # binary variable for b, u # User assignment
                self.mat_chi_action[e, :, t] = self.chi_action # a range between 0 and 1
                self.mat_b_connected_episodic[e, :, t] = self.chi_num # BS index that the user is connected to

                for u in range(USER_NO):
                    if t > 0:
                        if self.chi_num[u] != self.mat_b_connected_episodic[e, u, t-1]:
                            count_handovers += 1
                            self.mat_count_handovers[e, u] += 1
                
                self.mat_rho[e,:,:,:,t] = self.rho
                self.mat_rho_action[e, :, :, t] = self.rho_action # a range between 0 and 1
                # self.mat_satisfied_prb_constraint[e, t] = prb_cnt_u / USER_NO

                PRB_utilization = np.zeros([BS_NO])
                # Calculate PRB utilization for each base station
                for b in range(BS_NO):
                    PRB_utilization[b] = np.sum(self.rho[b, :, :]) / PRB_NO
                #maybe also give negative reward if avg utilization is low
                self.mat_prb_util_per_bs[e, :, t] = PRB_utilization
                self.prb_util_t[e,:,t] = PRB_utilization

                # for u in range(USER_NO):
                #     for b in range(BS_NO):
                #         if self.chi[u, b] == 1:
                #             self.mat_used_prbs_per_user[e, :, t] = np.sum(self.rho[b, :, u])
                # self.prb_ratio_t[e,:,t] = self.mat_used_prbs_per_user[e, :, t] / PRB_NO
                self.prb_ratio_t[e,:,t] = self.rho_num / PRB_NO
                #p
                # self.mat_p_compressed[e, :, t] = self.p_compressed # between 0 and 1
                self.mat_p[e, :, :, :, t] = self.p
                self.mat_p_action[e, :, :, t] = self.p_action # a range between 0 and 1

                self.mat_sum_power[e, :, t] = self.p_num

                power_utilization = np.zeros([BS_NO])
                # Calculate power utilization for each base station
                for b in range(BS_NO):
                    power_utilization[b] = np.sum(self.p[b, :, :]) / MAX_POWER
                
                self.power_util_t[e,:,t] = power_utilization
                self.power_ratio_t[e,:,t] = self.p_num / MAX_POWER
                
                # self.mat_satisfied_power_constraint[e, t] = power_cnt_u / USER_NO
                #################

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

                ##Reward with number of connected devices! (ANOTHER IDEA)
                self.sigma_SSL_R = 0
                cnt_rate_passed_u = 0
                for s in range(SLICE_NO):
                    for u in range(USER_NO):
                        if self.mat_specs[u, 0] == s:
                            # min_rate specification
                            self.R_s = self.mat_specs[u, 1]
                            self.mat_ssl_u_rate[e, u, t] = (self.mat_rate[u] / self.R_s)
                            temp_rate_satisfaction_ratio = (self.mat_rate[u] / self.R_s)**XI # to widen the gap between satisfied and unsatisfied users
                            self.mat_fittingness_u_rate[e, u, t] = (temp_rate_satisfaction_ratio) / (1 + temp_rate_satisfaction_ratio) # sigmoid function
                            cnt_rate_passed_u += (self.mat_rate[u] >= self.R_s)
                            if self.mat_fittingness_u_rate[e, u, t] < 0.5 :# self.mat_rate[u] < self.R_s:
                                self.mat_reward_user[e, u, t] -= (0.5 - self.mat_fittingness_u_rate[e, u, t]) 
                
                self.mat_satisfied_rate_constraint[e, t] = cnt_rate_passed_u / USER_NO
                ##########################
                self.sigma_SSL_D = 0
                cnt_delay_passed_u = 0
                for s in range(SLICE_NO):
                    for u in range(USER_NO):
                        if self.mat_specs[u, 0] == s:
                            # max_tolerable_delay specification
                            self.D_s = self.mat_specs[u, 2]
                            self.mat_ssl_u_delay[e, u, t] = (self.D_s / self.mat_delay_tot[u])
                            temp_delay_satisfaction_ratio = (self.D_s / self.mat_delay_tot[u])**XI # to widen the gap between satisfied and unsatisfied users
                            self.mat_fittingness_u_delay[e, u, t] = (temp_delay_satisfaction_ratio) / (1 + temp_delay_satisfaction_ratio) # sigmoid function
                            cnt_delay_passed_u += (self.mat_delay_tot[u] <= self.D_s)
                            if self.mat_fittingness_u_delay[e, u, t] < 0.5 :
                                self.mat_reward_user[e, u, t] -= (0.5 - self.mat_fittingness_u_delay[e, u, t])
                
                self.mat_fittingness_u_delay[e, :, t] = np.clip(self.mat_fittingness_u_delay[e,:, t], 1e-10, 1.0)
                self.mat_fittingness_u_rate[e, :, t] = np.clip(self.mat_fittingness_u_rate[e,:, t], 1e-10, 1.0)
                for u in range(USER_NO):
                    self.mat_ssl_u_total[e, u, t] = ((self.mat_fittingness_u_rate[e, u, t])**(OMEGA_1)) * ((self.mat_fittingness_u_delay[e, u, t])**(1 - OMEGA_1)) # utility function
                    # if self.mat_reward_user[e, u, t] < 0:
                    #     self.reward += 100 * self.mat_reward_user[e, u, t] / USER_NO

                self.mat_satisfied_delay_constraint[e, t] = cnt_delay_passed_u / USER_NO

                self.mat_ssl[e, t] = np.prod([self.mat_ssl_u_total[e, u, t] for u in range(USER_NO)]) ** (1 / USER_NO)
                self.mat_ssl_rate[e, t] = np.prod([self.mat_fittingness_u_rate[e, u, t] for u in range(USER_NO)]) ** (1 / USER_NO)
                self.mat_ssl_delay[e, t] = np.prod([self.mat_fittingness_u_delay[e, u, t] for u in range(USER_NO)]) ** (1 / USER_NO)

                ####NEW REWARD MECHANISM 
                if self.mat_ssl[e, t] == 0:
                    print('error')
                
                if self.mat_ssl[e, t] < 1e-10:
                    print('-10')
                self.mat_ssl_u_total[e, :, t] = np.clip(self.mat_ssl_u_total[e, :, t], 1e-10, 1.0) #ensuring that we do not have zero SSLs to avoid division by zero in the logarithmic function
                self.mat_ssl[e, t] = np.clip(self.mat_ssl[e, t], 1e-10, 1.0)


                
                    # else:
                    #     print('DEBUG ME')

                # Logarithmic transformation of the geometric mean (SSL)
                                # self.log_geometric_mean[e, t] = np.mean(np.log(self.mat_ssl_u_total[e, :, t]))
                

                # self.log_geometric_mean[e, t] = np.log(self.mat_ssl[e, t])
                # Regularization term to penalize low satisfaction levels (it's like the sum of squared errors)
                # self.regularization_term[e, t] = IOTA * np.sum((1 - self.mat_ssl_u_total[e, :, t]) ** 2)
                # self.logarithmic_reward[e, t] = self.log_geometric_mean[e, t] - self.regularization_term[e, t] # self.reward += np.sum(self.mat_rate) ############# CHANGED the reward to sum rate # self.reward += self.logarithmic_reward[e, t] 
                self.logarithmic_reward[e, t] = np.log(self.mat_ssl[e, t])
                ##Counting the number of happy users## 
                self.mat_no_of_satisfied_delay_and_rate_constraint[e, t] = cnt_delay_passed_u + cnt_rate_passed_u #2U
                           
                # self.reward = self.mat_no_of_satisfied_delay_and_rate_constraint[e, t] / (2*USER_NO)
                
                # self.reward = self.mat_no_of_satisfied_delay_and_rate_constraint[e, t] * self.mat_ssl[e, t] / (2*USER_NO)
                # self.reward += np.log(self.mat_ssl[e, t])
                # self.reward = np.average(self.mat_ssl_u_total[e, :, t])

                # self.reward = self.mat_ssl_rate[e, t]
                # self.reward = np.average(self.mat_fittingness_u_rate[e, :, t])

                # self.reward += np.sum(self.mat_rate) # self.log_geometric_mean[e, t]  
                # if cnt_delay_passed_u == USER_NO:
                #     if cnt_rate_passed_u == USER_NO:
                #         #maybe save these!
                #         print(style.RED + 'NICE! Episode: {}, Timestep: {}, Reward: {}'.format(e, t, self.reward))
                
                # if self.mat_ssl[e,t] >= 0.5:
                #     self.reward = self.mat_ssl[e, t]
                self.total_user_assignment_penalty[e,t], self.user_assignment_penalty_per_user[e,:,t] = self.calculate_penalty()                

                self.reward = self.mat_ssl[e, t] #- self.total_user_assignment_penalty[e,t]
                # self.reward = - self.total_user_assignment_penalty[e,t]
                    
                
                
                
                self.mat_reward[e, t] = self.reward

                # if cnt_delay_passed_u == USER_NO:
                #     if cnt_rate_passed_u == USER_NO:
                #         # self.reward += 100 * self.mat_ssl[e, t] # very positive reward
                #         print(style.RED + 'NICE! Episode: {}, Timestep: {}, Reward: {}'.format(e, t, self.reward))

                # # ---------Next state calculation--------------
                LC_next = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY,
                              X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY)
                self.loc_users_new, self.H_new, self.mat_distance_new, self.angle_next = LC_next.user_location(self.tt, self.loc_user, self.mat_bs_loc, self.angle_historic[e,:,t])
                self.angle_historic[e,:,self.tt] = self.angle_next

                # self.H_b_new = np.zeros([BS_NO, USER_NO])
                # self.H_b_new_normalized = np.zeros([BS_NO, USER_NO])
                # for u in range(USER_NO):
                #     for b in range(BS_NO):
                #         self.H_b_new[b, u] = np.average(self.H_new[b, :, u])
                # min_val = np.min(self.H_b_new)
                # max_val = np.max(self.H_b_new)

                # self.H_b_new_normalized = (self.H_b_new - min_val) / (max_val - min_val)
                # -----------------------------------------------------
                SC = StateCalculation(self.mat_ssl_u_total[e, :, t], self.prb_util_t[e, :, t], self.prb_ratio_t[e, :, t], 
                                          self.power_util_t[e, :, t], self.power_ratio_t[e, :, t])                
                self.next_state = SC._()
                self.next_state = self.next_state.astype('float16')
                action_memory[t, :] = self.action
                # -------------------------------------
                self.agent.memorize(self.state, self.action, self.reward, self.next_state)
                self.agent.replay()
                # -------------------------------------------------------
                end_time = time.time()  # Record the end time of the loop                
                self.mat_episode_runtime[e,t] = end_time - start_time # Storing the episode/timeslot runtime duration in seconds
                # if e%200 == 0:
                #     if t%49 == 0:
                #         print('reward: ', self.reward)

                # # plot periodically:
                plt.clf() # Clear the current figure
                if t%300 == 0:
                    print('reward: ', self.reward)
                    WINDOW_SIZE = 10
                    data = [moving_average(self.mat_reward, WINDOW_SIZE)] # [m,t]
                    labels = ['SAC']
                    colors = ['b']  # choose colors for each curve
                    linestyles = ['-']  # choose line styles for each curve
                    plt.ion()  # Turn on interactive mode

                    plot_graph('Reward (Until timestep {}/{} of run {}/{})'.format(t, T, e, E), data, labels, colors, linestyles, "Timestep", "Reward")
            # in m loop
            if count_handovers>0:
                print(style.CYAN + 'Total Handovers  over all timesteps: Episode {}= {} HOs, reward= {}'.format(e, count_handovers, self.reward))
            if e%100 == 0:
                #print(style.CYAN + 'Total Handovers  over all timesteps: Episode {}= {} HOs, reward= {}'.format(e, count_handovers, self.reward))
                LC.plot_user_movement(self.loc_user, self.mat_chi[e, :, :, :], T-1)
        return self.mat_rho, self.mat_u_bs_dist, self.shannon, self.mat_gain, self.mat_p, self.mat_sum_power, self.mat_reward, self.mat_satisfied_prb_constraint, self.mat_satisfied_power_constraint, self.mat_satisfied_delay_constraint, self.mat_satisfied_rate_constraint, self.mat_ssl_rate, self.mat_ssl_delay, self.mat_ssl, self.mat_episode_runtime, self.mat_rate, self.monte_mat_delay_tot, self.mat_used_prbs_per_user, self.mat_prb_util_per_bs, self.du_ru_adj_matrix, LC, self.mat_chi, self.loc_user, self.max_rate, self.max_inversed_delay, self.mat_ssl_u_rate, self.mat_ssl_u_delay, self.mat_fittingness_u_rate, self.mat_fittingness_u_delay, self.mat_specs, self.mat_count_handovers, self.mat_ssl_u_total, self.mat_reward_user, self.logarithmic_reward, self.mat_no_of_satisfied_delay_and_rate_constraint, self.total_user_assignment_penalty, self.user_assignment_penalty_per_user, self.prb_util_t, self.power_util_t, self.prb_ratio_t, self.power_ratio_t, self.mat_chi_action, self.mat_p_action, self.mat_rho_action


# %%
M = _main_(E, T)
mat_rho, mat_u_bs_dist, shannon, mat_gain, mat_p, mat_sum_power, mat_reward, mat_satisfied_prb_constraint, mat_satisfied_power_constraint, mat_satisfied_delay_constraint, mat_satisfied_rate_constraint, mat_ssl_rate, mat_ssl_delay, mat_ssl, mat_episode_runtime, mat_rate, monte_mat_delay_tot, mat_used_prbs_per_user, mat_prb_util_per_bs, du_ru_adj_matrix, LC, mat_chi, loc_user, max_rate, max_inversed_delay, mat_ssl_u_rate, mat_ssl_u_delay, mat_fittingness_u_rate, mat_fittingness_u_delay, mat_specs, mat_count_handovers, mat_ssl_u_total, mat_reward_user, logarithmic_reward, mat_no_of_satisfied_delay_and_rate_constraint, total_user_assignment_penalty, user_assignment_penalty_per_user, prb_util_t, power_util_t, prb_ratio_t, power_ratio_t, mat_chi_action, mat_p_action, mat_rho_action = M._()

# %%%%%%%

# save for later (use savez_compressed for compression)
filename = f'O-RAN SAC, U={USER_NO}, PRB={PRB_NO}, BS={BS_NO}, E={E}, T={T}, VELOCITY={VELOCITY}, D_max={CONST_D_MAX}, R_min={CONST_R_MIN}.npz'
np.savez_compressed(filename, mat_rho=mat_rho, mat_u_bs_dist=mat_u_bs_dist, shannon=shannon, mat_gain=mat_gain, mat_p=mat_p, mat_sum_power=mat_sum_power, mat_reward=mat_reward, mat_satisfied_prb_constraint=mat_satisfied_prb_constraint, mat_satisfied_power_constraint=mat_satisfied_power_constraint,
                    mat_satisfied_delay_constraint=mat_satisfied_delay_constraint, mat_satisfied_rate_constraint=mat_satisfied_rate_constraint, mat_ssl_rate=mat_ssl_rate, mat_ssl_delay=mat_ssl_delay, mat_ssl=mat_ssl, mat_episode_runtime=mat_episode_runtime, mat_rate=mat_rate, 
                    monte_mat_delay_tot=monte_mat_delay_tot, mat_used_prbs_per_user=mat_used_prbs_per_user, mat_prb_util_per_bs=mat_prb_util_per_bs, du_ru_adj_matrix=du_ru_adj_matrix, mat_chi=mat_chi, max_rate=max_rate, max_inversed_delay=max_inversed_delay, 
                    mat_fittingness_u_rate=mat_fittingness_u_rate, mat_fittingness_u_delay=mat_fittingness_u_delay, mat_specs=mat_specs, mat_count_handovers=mat_count_handovers, mat_ssl_u_total=mat_ssl_u_total, mat_reward_user=mat_reward_user, logarithmic_reward=logarithmic_reward, 
                    mat_no_of_satisfied_delay_and_rate_constraint=mat_no_of_satisfied_delay_and_rate_constraint, total_user_assignment_penalty=total_user_assignment_penalty, user_assignment_penalty_per_user=user_assignment_penalty_per_user,
                    prb_util_t=prb_util_t, power_util_t=power_util_t, prb_ratio_t=prb_ratio_t, power_ratio_t=power_ratio_t, mat_chi_action=mat_chi_action, mat_p_action=mat_p_action, mat_rho_action=mat_rho_action)

#%% %PLOTTING THE RESULTS%%


window_size = 10  # (for smoothing the curves in the plots)
#####
LC.visualize_ru_du_locations(du_ru_adj_matrix)
# %%%%RUNTIME DURATION%%%%%%%
# Calculate the average episode runtime and its moving average
# mean_mat_episode_runtime = moving_average(1000*np.average(mat_episode_runtime, axis=1), window_size) 

# Plot the data using your custom plot function
plot_graph("Runtime Duration",
           [moving_average(1000*np.average(mat_episode_runtime, axis=0), window_size)],
           ['Average runtime duration: {:.2f} ms'.format(1000 * np.average(mat_episode_runtime))],
           ['blue'],
           ['solid'],
           "Time step",
           "Runtime duration (ms)")



#######################
# Calculate averages over Monte Carlo runs and users
avg_prbs_per_user = np.mean(prb_ratio_t, axis=(0,1))
# avg_prbs_per_user_pred = np.mean(mat_used_prbs_per_user_pred, axis=(0,1))

# Plot the averages using your function
plot_graph('Avg PRB ratio per user',
           [moving_average(avg_prbs_per_user , window_size)],
           ['SAC'],
           ['b'],
           ['-'],
           'Time step',
           'Average PRBs used per user')

avg_power_per_user = np.mean(power_ratio_t, axis=(0,1))
# avg_prbs_per_user_pred = np.mean(mat_used_prbs_per_user_pred, axis=(0,1))

# Plot the averages using your function
plot_graph('Avg power ratio per user',
           [moving_average(avg_power_per_user , window_size)],
           ['SAC'],
           ['b'],
           ['-'],
           'Time step',
           'Average power ratio (of RU) used per user')


avg_prbs_per_bs_util = np.mean(prb_util_t, axis=(0,1))
# avg_prbs_per_user_pred = np.mean(mat_used_prbs_per_user_pred, axis=(0,1))

# Plot the averages using your function
plot_graph('Avg PRB utilization per RU',
           [moving_average(avg_prbs_per_bs_util , window_size)],
           ['SAC'],
           ['b'],
           ['-'],
           'Time step',
           'Avg PRB utilization per RU')

avg_power_per_bs_util = np.mean(power_util_t, axis=(0,1))
# avg_prbs_per_user_pred = np.mean(mat_used_prbs_per_user_pred, axis=(0,1))

# Plot the averages using your function
plot_graph('Avg power utilization per RU',
           [moving_average(avg_power_per_user , window_size)],
           ['SAC'],
           ['b'],
           ['-'],
           'Time step',
           'Avg power utilization per RU')


group_size = T//10
num_groups = 10
# avg_prbs_per_user = np.mean(mat_used_prbs_per_user, axis=(1))
# temp_prb = avg_prbs_per_user[:,:-1]
# avg_prbs_per_user_box = np.average(temp_prb, axis=0)

# grouped_data = [avg_prbs_per_user_box[i * group_size:(i + 1) * group_size]for i in range(num_groups)]

# # Create a box plot for the grouped rewards

# plt.figure(figsize=(10, 6))
# plt.boxplot(grouped_data, vert=True, patch_artist=True)
# plt.xlabel('Groups of Time steps')
# plt.ylabel('Avg number of PRBs per user')
# # plt.title('Box Plot of PRB allocations Grouped by 50 Episodes')
# plt.xticks(ticks=np.arange(1, num_groups + 1), labels=[f'{i*group_size}-{(i+1)*group_size-1}' for i in range(num_groups)])
# plt.show()

# Fairness for PRBs
# Calculate PRB utilization for each user at each time step and for each MC run
# prb_utilization = np.sum(mat_rho, axis=(0, 1))  # Sum over PRBs and BSs

# # Calculate proportional fairness score for each time step
# fairness_scores = np.zeros((E, T))

# for e in range(E):
#     for t in range(T):
#         # Calculate Jain's fairness index for time step t in MC run mc
#         sum_of_prbs = np.sum(prb_utilization[e, :, t])
#         sum_of_squares = np.sum(prb_utilization[e, :, t] ** 2)
#         # Handle potential division by zero or NaN
#         if sum_of_prbs == 0 or np.isnan(sum_of_squares):
#             fairness_scores[e, t] = 0  # Set fairness score to 0
#         else:
#             fairness_scores[e, t] = (sum_of_prbs ** 2) / (BS_NO * sum_of_squares)

# # Average fairness scores over all MC runs
# avg_fairness_scores = np.mean(fairness_scores, axis=1)
# normalized_fairness_scores = (avg_fairness_scores - np.min(avg_fairness_scores)) / (np.max(avg_fairness_scores) - np.min(avg_fairness_scores))


# plot_graph("Jain's fairness index for PRB allocation",
#            [moving_average(normalized_fairness_scores , window_size)],
#            ['SAC'],
#            ['blue'],
#            ['solid'],
#            "Episode",
#            "Mean fairness score")

# %%REWARD%%%%
plot_graph("User assignment penalty (not effective in solution)",
           [moving_average(np.average(total_user_assignment_penalty, axis=0), window_size)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Time step",
           "User assignment penalty")

plot_graph("logarithmic rewards",
           [moving_average(np.average(logarithmic_reward, axis=0), window_size)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Time step",
           "logarithmic rewards")
#e,t
plot_graph("Mean episodic rewards",
           [moving_average(np.average(mat_reward, axis=0), window_size)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Time step",
           "Mean episodic rewards")

mat_reward_temp = mat_reward[:,:-1]
mat_reward_boxplot = np.average(mat_reward_temp, axis=0)
data = [mat_reward_boxplot[i*group_size:(i+1)*group_size] for i in range(num_groups)]

# Plotting
plt.figure(figsize=(12, 6))
plt.boxplot(data)
plt.xlabel('Groups of Time steps')
plt.ylabel('Reward')
plt.xticks(ticks=range(1, num_groups + 1), labels=[f'{i*group_size}-{(i+1)*group_size-1}' for i in range(num_groups)])
plt.show()


plot_graph("Constraint Satisfaction",
           [moving_average(np.average(mat_satisfied_delay_constraint, axis=0), window_size),
            moving_average(np.average(mat_satisfied_rate_constraint, axis=0), window_size)],
           ['Delay (SAC)',
            'Rate (SAC)'],
           ['red', 'blue'],
           ['solid', 'solid'],
           "Time step",
           "Constraint Satisfaction Rate")


temp_delay = mat_satisfied_delay_constraint[:, :-1]
mat_satisfied_delay_constraint_box = np.average(temp_delay, axis=0)
data = [mat_satisfied_delay_constraint_box[i*group_size:(i+1)*group_size] for i in range(num_groups)]

# Plotting
plt.figure(figsize=(12, 6))
plt.boxplot(data)
plt.xlabel('Groups of Time steps')
plt.ylabel('Constraint Satisfaction Rate (Delay)')
plt.xticks(ticks=range(1, num_groups + 1), labels=[f'{i*group_size}-{(i+1)*group_size-1}' for i in range(num_groups)])
plt.show()


temp_rate =mat_satisfied_rate_constraint[:, :-1]
mat_satisfied_rate_constraint_box = np.average(temp_rate, axis=0)
data = [mat_satisfied_rate_constraint_box[i*group_size:(i+1)*group_size] for i in range(num_groups)]

# Plotting
plt.figure(figsize=(12, 6))
plt.boxplot(data)
plt.xlabel('Groups of Time steps')
plt.ylabel('Constraint Satisfaction Rate (bitrate)')
plt.xticks(ticks=range(1, num_groups + 1), labels=[f'{i*group_size}-{(i+1)*group_size-1}' for i in range(num_groups)])
plt.show()


# %%%%%SSL%%%%%%
plot_graph("SSL Metrics",
           [moving_average(np.average(mat_ssl_rate, axis=0), window_size),
            moving_average(np.average(mat_ssl_delay, axis=0), window_size),
            moving_average(np.average(mat_ssl, axis=0), window_size)],
           ['Rate (SAC)',
            'Delay (SAC)',
            'SSL (SAC)'],
           ['blue', 'green', 'orange'],
           ['solid', 'solid', 'solid'],
           "Time step",
           "SSL Metrics")

temp_delay = mat_ssl_delay[:, :-1]
mat_ssl_delay_box = np.average(temp_delay, axis=0)
data = [mat_ssl_delay_box[i*group_size:(i+1)*group_size] for i in range(num_groups)]

# Plotting
plt.figure(figsize=(12, 6))
plt.boxplot(data)
plt.xlabel('Groups of Time steps')
plt.ylabel('Delay SSL')
plt.xticks(ticks=range(1, num_groups + 1), labels=[f'{i*group_size}-{(i+1)*group_size-1}' for i in range(num_groups)])
plt.show()


temp_rate =mat_ssl_rate[:, :-1]
mat_ssl_rate_box = np.average(temp_rate, axis=0)
data = [mat_ssl_rate_box[i*group_size:(i+1)*group_size] for i in range(num_groups)]

# Plotting
plt.figure(figsize=(12, 6))
plt.boxplot(data)
plt.xlabel('Groups of Time steps')
plt.ylabel('Bitrate SSL')
plt.xticks(ticks=range(1, num_groups + 1), labels=[f'{i*group_size}-{(i+1)*group_size-1}' for i in range(num_groups)])
plt.show()



temp_ssl =mat_ssl[:, :-1]
mat_ssl_box = np.average(temp_ssl, axis=0)
data = [mat_ssl_box[i*group_size:(i+1)*group_size] for i in range(num_groups)]

# Plotting
plt.figure(figsize=(12, 6))
plt.boxplot(data)
plt.xlabel('Groups of Time steps')
plt.ylabel('Total SSL')
plt.xticks(ticks=range(1, num_groups + 1), labels=[f'{i*group_size}-{(i+1)*group_size-1}' for i in range(num_groups)])
plt.show()

# %%%%%Delay%%%%%%
# Calculate mean delay over users for SAC
mean_delay_sac = np.mean(np.mean(monte_mat_delay_tot, axis=1), axis=0)
# Plot the comparison graph
plot_graph("Comparison of Average E2E Delay (SAC)",
           [moving_average(mean_delay_sac , window_size)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Time step",
           "Average E2E Delay (ms)")

#####################
mean_rate_sac = np.mean(np.mean(shannon, axis=1), axis=0)
# mean_rate_sac_smoothed = moving_average(mean_rate_sac, window_size)
plot_graph("Average Data Rate (SAC)",
           [moving_average(mean_rate_sac , window_size)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Time step",
           "Average Data Rate (Mbps)")

#####################
mean_power_sac = np.mean(np.mean(mat_sum_power, axis=1), axis=0)
# mean_power_sac_smoothed = moving_average(mean_power_sac, window_size)
plot_graph("Average of total allocated power to each user (SAC)",
           [moving_average(mean_power_sac , window_size)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Time step",
           "Average of total allocated power to each user (W)")

###
average_rate = np.mean(shannon, axis=0)

for user_idx in range(USER_NO):
    plt.plot(moving_average(range(T), window_size), moving_average(average_rate[user_idx,:], window_size), label=f'User {user_idx+1}')
plt.xlabel('Time step')
plt.ylabel('Rate (Mbps)')
plt.title('Average rate for individual users')
plt.legend()
plt.show()

###
average_rate_ssl_u = np.mean(mat_fittingness_u_rate, axis=0)

for user_idx in range(USER_NO):
    plt.plot(moving_average(range(T), window_size), moving_average(average_rate_ssl_u[user_idx,:], window_size), label=f'User {user_idx+1}')
plt.xlabel('Time step')
plt.ylabel('SSL_rate_u')
plt.title('Normalized rate satisfaction')
plt.legend()
plt.show()

###
average_delay_ssl_u = np.mean(mat_fittingness_u_delay, axis=0)

for user_idx in range(USER_NO):
    plt.plot(moving_average(range(T), window_size), moving_average(average_delay_ssl_u[user_idx,:], window_size), label=f'User {user_idx+1}')
plt.xlabel('Time step')
plt.ylabel('SSL_delay_u')
plt.title('Normalized delay satisfaction')
plt.legend()
plt.show()
###
plot_graph("Sum of handovers ",
           [moving_average(np.sum(mat_count_handovers, axis=1) , 1)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Episode",
           "Sum of handovers")

####
plot_graph("No. of satisfied delay and rate constraints (Max= 2*U)",
           [moving_average(np.average(mat_no_of_satisfied_delay_and_rate_constraint, axis=0), window_size)],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Timestep",
           "No. of satisfied requirements")
###

print(style.UNDERLINE + "Total time for {} timesteps ({} users) in {} Episdoes (aka iterations or epochs): {}".format(T, USER_NO, E, convert_seconds(np.sum(mat_episode_runtime))))

