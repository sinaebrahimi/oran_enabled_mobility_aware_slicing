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
PACKET_NO = config['PACKET_NO']
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
        SP = Specifications(USER_NO, SLICE_NO, CONST_D_MAX, CONST_R_MIN, PACKET_SIZE, PACKET_NO)
        self.mat_specs = SP._()
        # --------------------------------------
        self.loc_user_init = np.zeros([T, USER_NO, 2]) # initializing user_location... t=0 location will be changed randomly in the RadioCalc.user_location()
        #self.mat_loc_user = np.zeros([MC, T, USER_NO, 2])
        # ---------
        RLOC_INIT = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY, X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY)
        self.mat_fh_links_capacity, self.mat_e2_links_capacity = RLOC_INIT.links_capacity()        
        # -----------------------------------------------
        # self.mat_reward = np.zeros([T])
        self.mat_reward = np.zeros([MC, T]) # -100 * np.ones([MC, T])
        self.mat_satisfied_prb_constraint = np.zeros([MC, T])
        self.mat_satisfied_power_constraint = np.zeros([MC, T])
        self.mat_satisfied_rate_constraint = np.zeros([MC, T])
        self.mat_satisfied_delay_constraint = np.zeros([MC, T])
        # self.mat_satisfied_fh_link_capacity_constraint = np.zeros([MC, T])
        # self.mat_satisfied_e2_link_capacity_constraint = np.zeros([MC, T])
        self.mat_ssl_u_rate = np.zeros([MC, USER_NO, T])
        self.mat_ssl_u_delay = np.zeros([MC, USER_NO, T])
        self.mat_ssl_rate = np.zeros([MC, T])
        self.mat_ssl_delay = np.zeros([MC, T])
        self.mat_ssl = np.zeros([MC, T])
        self.mat_episode_runtime = np.zeros([MC, T])
        self.mat_associator = np.zeros([USER_NO, BS_NO, T]) # only get the latest MC

        self.shannon = np.zeros([MC, USER_NO, T])

        self.mat_power = np.zeros([MC, USER_NO, T])
        self.mat_gain = np.zeros([MC, USER_NO, PRB_NO, T])
        self.mat_rho  = np.zeros([MC, BS_NO, PRB_NO, USER_NO, T])
        self.mat_u_bs_dist = np.zeros([MC, USER_NO, T])
        # -------------
        # Initialize matrices for SAC
        self.mat_used_prbs_per_user_per_bs = np.zeros((MC, BS_NO, USER_NO, T))
        self.mat_used_prbs_per_user = np.zeros((MC, USER_NO, T))
        self.mat_b_connected_episodic = np.zeros((MC, USER_NO, T))

        # # Initialize matrices for SAC_pred
        # self.mat_used_prbs_per_user_per_bs_pred = np.zeros((MC, BS_NO, USER_NO, T))
        # self.mat_used_prbs_per_user_pred = np.zeros((MC, USER_NO, T))
        # #---------
        # self.mat_reward_pred = -100 * np.ones([MC, T])
        # self.mat_satisfied_prb_constraint_pred = np.zeros([MC, T])
        # self.mat_satisfied_power_constraint_pred = np.zeros([MC, T])e
        # self.mat_satisfied_delay_constraint_pred = np.zeros([MC, T])
        # # self.mat_satisfied_fh_link_capacity_constraint_pred = np.zeros([MC, T])
        # # self.mat_satisfied_e2_link_capacity_constraint_pred = np.zeros([MC, T])
        # self.mat_ssl_u_rate_pred = np.zeros([MC, USER_NO, T])
        # self.mat_ssl_u_delay_pred = np.zeros([MC, USER_NO, T])
        # self.mat_ssl_rate_pred = np.zeros([MC, T])
        # self.mat_ssl_delay_pred = np.zeros([MC, T])
        # self.mat_ssl_pred = np.zeros([MC, T])

        # self.shannon_pred = np.zeros([MC, USER_NO, T])

        # self.mat_power_pred = np.zeros([MC, USER_NO, T])
        # self.mat_gain_pred = np.zeros([MC,-100 * np.ones([MC, T]) USER_NO, PRB_NO, T])
        # self.mat_u_bs_dist_pred = np.zeros([MC, USER_NO, T])
        # ---------------------------------------------------------
        #self.monte_mat_rate = np.zeros([MC,-100 * np.ones([MC, T]) T, USER_NO])
        #self.list_rate = []
        # ---------------------------------------------------------
        self.monte_mat_delay_tot = np.zeros([MC, USER_NO, T])
#        self.list_delay = []
        # ---------
        #self.monte_mat_rate_pred = np.zeros([MC, T, USER_NO])
        #self.list_rate_pred = []
        # ---------------------------------------------------------
        #self.monte_mat_delay_tot_pred = np.zeros([MC, USER_NO, T])
        # ----------obtaining the number of actions--------------
        self.e1 = BS_NO * PRB_NO * USER_NO # ran_prb_allocation()
        self.e2 = self.e1 + BS_NO * PRB_NO * USER_NO # ran_power_allocation()
        self.num_actions = self.e2 # assuming that the user_association is conducted using a heuristic algorithm based on min_distance in user_location(self, t, loc_user)
        # ---------------------------------------------------------e
        self.s1 = 2 * USER_NO # np.zeros([T, USER_NO, 2]) ## 2 for x,y of t+1
        self.s2 = USER_NO * BS_NO * PRB_NO # channel gain matrix (b,k,u) of t+1 # self.H = np.zeros([self.BS_NO, self.PRB_NO, self.USER_NO]) # defined in radio_calc.py -> user_location()
        # add the occupancy/availability of PRBs to the state: rho[b,k,u] of t (current real state (unpredicted)); (we should know which are not being utilized)
        # add the remaining power capacity of RUs to the state (unpredicted): [b] of t (it should be a vector like [40 22.5 30 40 5.5 30] for 6 RUs)
        # self.s3 = 4 * USER_NO #for slice requirements (but we dont need this as this wouldn't change!)
        self.state_size = self.s1 + self.s2

        # calling the SAC agent
        # Calling the SAC agent from sac_torch.py


    def _(self):
        LC = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY, X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY)
        # -------------------------------------
        self.mat_bs_loc = LC.bs_location()
        self.mat_du_loc = LC.du_location()
        self.distances_ric_du = LC.ric_du_distance()
        self.distances_du_ru = LC.du_ru_distance()
        self.du_ru_adj_matrix, self.ric_du_adj_matrix = LC.adj_matrix()
        for m in range(MC):
            #maybe reset the LSTM as well
            count_handovers = 0

            #loc_user_m = np.zeros([T, USER_NO, 2])

            #resetting the SAC agent here!            
            self.agent = Agent(ALPHA_ACT, BETA_ACT, self.num_actions, self.state_size)
            self.var = VAR # .9995 #experiment .9995 and .995 # can determine the ratio of exploration to exploitation
            self.decay_var = DECAY_VAR

            for t in range(T-1):
                start_time = time.time()
                self.tt = t + 1
                if np.mod(t, 1000) == 0:
                    print(style.RED + str(t))
                # starting with a negative award, aiming to learn more in initial episodes; Not sure if it is necessary (due to line 495)
                self.reward = 0 # -100
                # -----------------------------------------------                
                if t == 0:
                    self.loc_user = self.loc_user_init
                self.loc_user, self.H, self.associator, self.mat_distance, self.mat_b_connected = LC.user_location(t, self.loc_user, self.mat_bs_loc)
                self.mat_associator[:, :, t] = self.associator
                #print(self.loc_user[t,0,:])
                #print('-')
                self.mat_b_connected_episodic[m, :, t] = self.mat_b_connected

                # Find the indices where associator is 1
                u_indices, b_indices = np.where(self.associator == 1)
                for u in range(USER_NO):
                    if t > 0:
                        if self.mat_b_connected[u] != self.mat_b_connected_episodic[m, u, t-1]:
                            count_handovers += 1
                            print(style.BLUE + 'Handover in timestep {} for user {}: from RU {} to RU {}'.format(t, u, self.mat_b_connected_episodic[m, u, t-1], self.mat_b_connected[u]))

                # Use these indices to directly assign values
                self.mat_u_bs_dist[m, u_indices, t] = self.mat_distance[b_indices, u_indices]
                self.mat_gain[m, u_indices, :, t] = self.H[b_indices, :, u_indices]

                # for u in range(USER_NO):
                #     for b in range(BS_NO):
                #         if self.associator[u, b] == 1:
                #             self.mat_u_bs_dist[m, u, t] = self.mat_distance[b, u]
                #             #self.mat_u_bs_dist_pred[m, u, t] = self.mat_distance_pred[b, u]

                #             self.mat_gain[m, u, :, t] = self.H[b, :, u]  # b,k,u
                #             #self.mat_gain_pred[m, u, :, t] = (self.H_pred[b, :, u])  # b,k,u

                # -----------------------------------------------------
                SC = StateCalculation(self.loc_user[t, :], self.H, X_LIM)
                self.state = SC._()
                self.mat_delay_tot = np.ones([USER_NO])
                #self.mat_delay_tot_pred = np.ones([USER_NO])
                self.mat_rate = np.zeros([USER_NO])
                #self.mat_rate_pred = np.zeros([USER_NO])
                # -----------------------------------------------------
                self.var = self.var * self.decay_var
                self.noise = np.random.randn(self.num_actions)
                self.noise = self.noise * self.var
                self.action = self.agent.choose_action(self.state)  # Choosing the action
                self.action += self.noise
                self.action = np.clip(self.action, -1, 1)
                # -------Current state calculation---------------------
                MA = Mapping(self.action, self.mat_specs, self.associator, USER_NO, BS_NO, PRB_NO, MAX_POWER)
                # self.done_user_prb_allocation, self.rho = MA.ran_prb_allocation()
                self.done_user_prb_allocation, self.rho, prb_cnt_u = MA.ran_prb_allocation() ####GREEDY VERSION ran_prb_allocation_greedy_version
                self.mat_satisfied_prb_constraint[m, t] = prb_cnt_u / USER_NO # ratio of the users with satisfied PRB allocation
                if self.mat_satisfied_prb_constraint[m, t] > 0.8: #if self.done_user_prb_allocation == 1: 
                    self.mat_rho[m, :, :, :, t] = self.rho #saving rho
                    #self.mat_satisfied_prb_constraint[m, t] = 1
                    self.done_user_power_allocation, self.P, power_cnt_u = MA.ran_power_allocation()
                    # ################################################
                    # ###### Make power alloc greedy for user=0
                    # selected_bs = self.mat_b_connected[0].astype(int)
                    # self.P[selected_bs,:,0] = MAX_POWER / PRB_NO 
                    # ######
                    # ###############Make PRB alloc GREEDY for user=0                    
                    #self.rho[selected_bs,:,0] = 1 # user=0
                    # #################################################
                    for u in range(USER_NO):
                        self.mat_power[m, u, t] = np.sum(self.P[:, :, u]) # summing the power allocated to all user PRBs
                    
                    self.mat_satisfied_power_constraint[m, t] = power_cnt_u / USER_NO # ratio of the users with satisfied power allocation
                    if self.mat_satisfied_power_constraint[m, t] >0.8: #if self.done_user_power_allocation == 1:
                        #self.mat_satisfied_power_constraint[m, t] = 1
                        
                        RC = RateCalculation(self.P, self.rho, self.H, self.associator, BS_NO, PRB_NO, USER_NO, SIGMA_NOISE, BW)
                        self.mat_rate, self.mat_rate_prb, self.SINR_dB, self.signal_strength_dB, self.interference_dB, self.noise_plus_interference_dB, self.used_prbs_per_user_per_bs, self.num_prbs_used_per_user = RC._()
                        self.mat_used_prbs_per_user_per_bs[m, :, :, t] = self.used_prbs_per_user_per_bs
                        self.mat_used_prbs_per_user[m, :, t] = self.num_prbs_used_per_user
                        self.shannon[m, :, t] = self.mat_rate  # b,k,u (in Mbps)   
                        cnt_rate_u = 0
                        for s in range(SLICE_NO):
                            for u in range(USER_NO):
                                if self.mat_specs[u, 0] == s:
                                    # min_rate specification
                                    self.R_s = self.mat_specs[u, 1]
                                    cnt_rate_u += (self.mat_rate[u] >= self.R_s)

                        self.mat_satisfied_rate_constraint[m, t] = cnt_rate_u / USER_NO
                        if self.mat_satisfied_rate_constraint[m, t] > 0.8: # or any other threshold!
                            #--------------------------------------
                            D = Delay(self.mat_rate, FH_BW_CAPACITY, E2_BW_CAPACITY, self.mat_specs, self.associator, 
                                    self.mat_distance, self.distances_ric_du, self.distances_du_ru, self.du_ru_adj_matrix, self.ric_du_adj_matrix, 
                                    USER_NO, BS_NO, DU_NO)
                            #self.mat_placement, self.W_link, self.mat_links_capacity, self.mat_nodes_and_vms_capacity, self.mat_specs, self.path, self.associator, self.mat_distance, USER_NO, VNF_NO, BS_NO)
                            cnt_u, done_delay_all,  self.mat_delay_tot = D._()
                            self.monte_mat_delay_tot[m, :, t] = self.mat_delay_tot                        
                            self.mat_satisfied_delay_constraint[m,t] = cnt_u / USER_NO
                            # -------------------------------------
                            #if done_delay_all == 1:
                            if self.mat_satisfied_delay_constraint[m,t] > 0.8: # or any other threshold!

                                ##Reward with number of connected devices!
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
                                    self.reward += self.mat_ssl[m, t] # between 0 and 1 ###100 * self.mat_ssl[m, t] # between 0 and 100
                                    # print(style.GREEN + 'Reward: {} in episode {} MC {}'.format(self.reward, t, m))
                                    self.mat_reward[m, t] = self.reward
                                else:
                                    print(style.YELLOW + '(Unsatisfied) Reward: {} in episode {} MC {}'.format(self.reward, t, m))
                                #############Reward with sum rate########################
                                # for u in range(USER_NO):
                                #     self.reward += self.mat_rate[u]
                                #     self.mat_reward[m, t] = self.reward


                                ######################
                                
                                
                                #else:
                                #    self.reward = 0
                            else:
                                self.reward = self.mat_satisfied_delay_constraint[m,t] - 1 #between -1 and 0 (If the satisfaction is 0.7, reward would be -0.3)
                                self.mat_reward[m, t] = self.reward
                        else:
                            self.reward = self.mat_satisfied_rate_constraint[m,t] - 1 #between -1 and 0 (If the satisfaction is 0.7, reward would be -0.3)
                            self.mat_reward[m, t] = self.reward
                    else:
                        self.reward = self.mat_satisfied_power_constraint[m, t] - 1 #between -1 and 0 (If the satisfaction is 0.7, reward would be -0.3)
                        self.mat_reward[m, t] = self.reward
                else:
                    self.reward = self.mat_satisfied_prb_constraint[m, t] - 1 #between -1 and 0 (If the satisfaction is 0.7, reward would be -0.3)
                    self.mat_reward[m, t] = self.reward

                # #########################################
                # # ------Proactive calculation--------
                # # -----------------------------------------------------
                # self.reward_pred = -100
                # # loc_user is calculated inside LC.user_location (H_pred is its result)
                # SC_pred = StateCalculation(self.H_pred, self.loc_user_pred[t, :])
                # self.state_pred = SC_pred._()

                # ###
                # self.action_pred = self.agent.choose_action(self.state_pred)  # Choosing the action
                # self.action_pred += self.noise
                # self.action_pred = np.clip(self.action_pred, -1, 1)
                # ##################
                # MA_pred = Mapping(self.action_pred, self.mat_specs, self.associator_pred, USER_NO, BS_NO, PRB_NO, MAX_POWER)
                # self.done_user_prb_allocation_pred, self.rho_pred = MA_pred.ran_prb_allocation()
                # if self.done_user_prb_allocation_pred == 1:
                #     self.mat_satisfied_prb_constraint_pred[m, t] = 1
                #     self.done_user_power_allocation_pred, self.P_pred = MA_pred.ran_power_allocation()
                #     for u in range(USER_NO):
                #         self.mat_power_pred[m, u, t] = np.sum(self.P_pred[:, :, u])
                #     if self.done_user_power_allocation_pred == 1:
                #         self.mat_satisfied_power_constraint_pred[m, t] = 1
                #         RC_pred = RateCalculation(self.P_pred, self.rho_pred, self.H_pred, self.associator_pred, BS_NO, PRB_NO, USER_NO, SIGMA_NOISE, BW)
                #         self.mat_rate_pred, self.mat_rate_prb_pred, self.SINR_dB_pred, self.signal_strength_dB_pred, self.interference_dB_pred, self.noise_plus_interference_dB_pred, self.used_prbs_per_user_per_bs_pred, self.num_prbs_used_per_user_pred  = RC_pred._() # m, u, t
                #         self.mat_used_prbs_per_user_per_bs_pred[m, :, :, t] = self.used_prbs_per_user_per_bs_pred
                #         self.mat_used_prbs_per_user_pred[m, :, t] = self.num_prbs_used_per_user_pred
                #         self.shannon_pred[m, :, t] = self.mat_rate_pred
                #         #--------------------------------------
                #         D_pred = Delay(self.mat_rate_pred, FH_BW_CAPACITY, E2_BW_CAPACITY, self.mat_specs, self.associator_pred, 
                #                   self.mat_distance_pred, self.distances_ric_du, self.distances_du_ru, self.du_ru_adj_matrix, self.ric_du_adj_matrix, 
                #                   USER_NO, BS_NO, DU_NO)
                #         cnt_u_pred, done_delay_all_pred,  self.mat_delay_tot_pred = D_pred._()
                #         self.monte_mat_delay_tot_pred[m, :, t] = self.mat_delay_tot_pred
                #         self.mat_satisfied_delay_constraint_pred[m,t] = cnt_u_pred / USER_NO
                #         # -------------------------------------
                #         self.monte_mat_delay_tot_pred[m, :, t] = self.mat_delay_tot_pred
                #         #done_delay_dummy = 1  # just tweaking. to not comment the next line
                #         if self.mat_satisfied_delay_constraint_pred[m,t] > 0.8:
                #             self.sigma_SSL_R_pred = 0
                #             for s in range(SLICE_NO):
                #                 for u in range(USER_NO):
                #                     if self.mat_specs[u, 0] == s:
                #                         # min_rate specification
                #                         self.R_s = self.mat_specs[u, 1]
                #                         self.mat_ssl_u_rate_pred[m, u, t] = (self.mat_rate_pred[u] / self.R_s)
                #                         self.sigma_SSL_R_pred += self.mat_ssl_u_rate_pred[m, u, t]

                #             self.SSL_R_pred = self.sigma_SSL_R_pred / (1 + self.sigma_SSL_R_pred)
                #             self.mat_ssl_rate_pred[m, t] = self.SSL_R_pred

                #             self.sigma_SSL_D_pred = 0
                #             for s in range(SLICE_NO):
                #                 for u in range(USER_NO):
                #                     if self.mat_specs[u, 0] == s:
                #                         # max_tolerable_delay specification
                #                         self.D_s = self.mat_specs[u, 2]
                #                         self.mat_ssl_u_delay_pred[m, u, t] = (self.D_s / self.mat_delay_tot_pred[u])
                #                         self.sigma_SSL_D_pred += self.mat_ssl_u_delay_pred[m, u, t]

                #             self.SSL_D_pred = self.sigma_SSL_D_pred / (1 + self.sigma_SSL_D_pred)
                #             self.mat_ssl_delay_pred[m, t] = self.SSL_D_pred
                            
                #             # -------------------
                #             self.mat_ssl_pred[m, t] = (self.SSL_R_pred**(OMEGA_1)) * ((self.SSL_D_pred)**(1 - OMEGA_1))

                #             if self.mat_ssl_pred[m, t] >= 0.5:
                #                 # C10 constraint
                #                 self.reward_pred += 100 * self.mat_ssl_pred[m, t]
                #                 print(style.BLUE + 'Reward (Proactive): {} in episode {} MC {}'.format(self.reward_pred, t, m))
                #                 self.mat_reward_pred[m, t] = self.reward_pred

                ##############################
                # ---------Next state calculation--------------
                LC_next = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY,
                              X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY)
                self.loc_users_new, self.H_new, self.associator, self.mat_distance, self.mat_b_connected = LC_next.user_location(self.tt, self.loc_user, self.mat_bs_loc)
                # -----------------------------------------------------
                SC = StateCalculation(self.loc_users_new[self.tt, :], self.H_new, X_LIM) #  shouldn't it be self.loc_users_new[self.tt, :]?
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

                # plot periodically:
                plt.clf() # Clear the current figure
                if t%1000 == 0:
                    WINDOW_SIZE = 200
                    data = [moving_average(self.mat_reward, WINDOW_SIZE)] # [m,t]
                    labels = ['SAC']
                    colors = ['b']  # choose colors for each curve
                    linestyles = ['-']  # choose line styles for each curve
                    plt.ion()  # Turn on interactive mode

                    plot_graph('Reward (Until episode {}/{} of run {}/{})'.format(t, T, m, MC), data, labels, colors, linestyles, "Episode", "Episodic Reward")
            # in m loop
            print(style.CYAN + 'Total Handovers  over all timesteps: MC {}= {} HOs'.format(m, count_handovers))
            LC.plot_user_movement(self.loc_user, self.mat_associator, T-2)
        #return self.mat_rho, self.mat_u_bs_dist, self.mat_u_bs_dist_pred, self.shannon, self.shannon_pred, self.mat_gain, self.mat_gain_pred, self.mat_power, self.mat_power_pred, self.mat_reward, self.mat_reward_pred, self.mat_satisfied_prb_constraint, self.mat_satisfied_prb_constraint_pred, self.mat_satisfied_power_constraint, self.mat_satisfied_power_constraint_pred, self.mat_satisfied_delay_constraint, self.mat_satisfied_delay_constraint_pred, self.mat_ssl_rate, self.mat_ssl_rate_pred, self.mat_ssl_delay, self.mat_ssl_delay_pred, self.mat_ssl, self.mat_ssl_pred, self.mat_episode_runtime, self.mat_rate, self.mat_rate_pred, self.monte_mat_delay_tot, self.monte_mat_delay_tot_pred, self.mat_used_prbs_per_user, self.mat_used_prbs_per_user_per_bs, self.mat_used_prbs_per_user_pred, self.mat_used_prbs_per_user_per_bs_pred, self.du_ru_adj_matrix, LC, self.mat_associator, self.loc_user
        return self.mat_rho, self.mat_u_bs_dist, self.shannon, self.mat_gain, self.mat_power, self.mat_reward, self.mat_satisfied_prb_constraint, self.mat_satisfied_power_constraint, self.mat_satisfied_delay_constraint, self.mat_ssl_rate, self.mat_ssl_delay, self.mat_ssl, self.mat_episode_runtime, self.mat_rate, self.monte_mat_delay_tot, self.mat_used_prbs_per_user, self.mat_used_prbs_per_user_per_bs, self.du_ru_adj_matrix, LC, self.mat_associator, self.loc_user


# %%
M = _main_(MC, T)
mat_rho, mat_u_bs_dist, shannon, mat_gain, mat_power, mat_reward, mat_satisfied_prb_constraint, mat_satisfied_power_constraint, mat_satisfied_delay_constraint, mat_ssl_rate, mat_ssl_delay, mat_ssl, mat_episode_runtime, mat_rate, monte_mat_delay_tot, mat_used_prbs_per_user, mat_used_prbs_per_user_per_bs, du_ru_adj_matrix, LC, mat_associator, loc_user = M._()
# mat_rho, mat_u_bs_dist, mat_u_bs_dist_pred, shannon, shannon_pred, mat_gain, mat_gain_pred, mat_power, mat_power_pred, mat_reward, mat_reward_pred, mat_satisfied_prb_constraint, mat_satisfied_prb_constraint_pred, mat_satisfied_power_constraint, mat_satisfied_power_constraint_pred, mat_satisfied_delay_constraint, mat_satisfied_delay_constraint_pred, mat_ssl_rate, mat_ssl_rate_pred, mat_ssl_delay, mat_ssl_delay_pred, mat_ssl, mat_ssl_pred, mat_episode_runtime, mat_rate, mat_rate_pred, monte_mat_delay_tot, monte_mat_delay_tot_pred, mat_used_prbs_per_user, mat_used_prbs_per_user_per_bs, mat_used_prbs_per_user_pred, mat_used_prbs_per_user_per_bs_pred, du_ru_adj_matrix, LC, mat_associator, loc_user = M._()

# %%%%%%%

# save for later (use savez_compressed for compression)
filename = f'O-RAN SAC (Normal and Proactive), RAYLEIGH={RAYLEIGH_SCALE}, U={USER_NO}, PRB={PRB_NO}, T={T}, VELOCITY={VELOCITY}, OMEGA_1={OMEGA_1}, D_max={CONST_D_MAX}, R_min={CONST_R_MIN}.npz'
np.savez_compressed(filename, mat_rho=mat_rho, mat_u_bs_dist=mat_u_bs_dist, shannon=shannon, mat_gain=mat_gain, mat_power=mat_power, mat_reward=mat_reward, mat_satisfied_prb_constraint=mat_satisfied_prb_constraint, mat_satisfied_power_constraint=mat_satisfied_power_constraint,
                    mat_satisfied_delay_constraint=mat_satisfied_delay_constraint, mat_ssl_rate=mat_ssl_rate, mat_ssl_delay=mat_ssl_delay, mat_ssl=mat_ssl, mat_episode_runtime=mat_episode_runtime, mat_rate=mat_rate, monte_mat_delay_tot=monte_mat_delay_tot, mat_used_prbs_per_user=mat_used_prbs_per_user, mat_used_prbs_per_user_per_bs=mat_used_prbs_per_user_per_bs, du_ru_adj_matrix=du_ru_adj_matrix, mat_associator=mat_associator)
# np.savez_compressed(filename, mat_rho=mat_rho, mat_u_bs_dist=mat_u_bs_dist, mat_u_bs_dist_pred=mat_u_bs_dist_pred, shannon=shannon, shannon_pred=shannon_pred, mat_gain=mat_gain, mat_gain_pred=mat_gain_pred, mat_power=mat_power, mat_power_pred=mat_power_pred, mat_reward=mat_reward, mat_reward_pred=mat_reward_pred, mat_satisfied_prb_constraint=mat_satisfied_prb_constraint, mat_satisfied_prb_constraint_pred=mat_satisfied_prb_constraint_pred, mat_satisfied_power_constraint=mat_satisfied_power_constraint, mat_satisfied_power_constraint_pred=mat_satisfied_power_constraint_pred,
#                     mat_satisfied_delay_constraint=mat_satisfied_delay_constraint, mat_satisfied_delay_constraint_pred=mat_satisfied_delay_constraint_pred, mat_ssl_rate=mat_ssl_rate, mat_ssl_rate_pred=mat_ssl_rate_pred, mat_ssl_delay=mat_ssl_delay, mat_ssl_delay_pred=mat_ssl_delay_pred, mat_ssl=mat_ssl, mat_ssl_pred=mat_ssl_pred, mat_episode_runtime=mat_episode_runtime, mat_rate=mat_rate, mat_rate_pred=mat_rate_pred, monte_mat_delay_tot=monte_mat_delay_tot, monte_mat_delay_tot_pred=monte_mat_delay_tot_pred, mat_used_prbs_per_user=mat_used_prbs_per_user, mat_used_prbs_per_user_per_bs=mat_used_prbs_per_user_per_bs, mat_used_prbs_per_user_pred=mat_used_prbs_per_user_pred, mat_used_prbs_per_user_per_bs_pred=mat_used_prbs_per_user_per_bs_pred, du_ru_adj_matrix=du_ru_adj_matrix, mat_associator=mat_associator)

#%% %PLOTTING THE RESULTS%%
window_size = 200  # (for smoothing the curves in the plots)
#####
LC.visualize_ru_du_locations(du_ru_adj_matrix)
# %%%%RUNTIME DURATION%%%%%%%
# Calculate the average episode runtime and its moving average
mean_mat_episode_runtime = moving_average(1000*np.average(mat_episode_runtime, axis=0), window_size)

# Plot the data using your custom plot function
plot_graph("Runtime Duration",
           [mean_mat_episode_runtime],
           ['Average runtime duration: {:.2f} ms'.format(1000 * np.average(mat_episode_runtime))],
           ['blue'],
           ['solid'],
           "Episode",
           "Runtime duration (ms)")
#PRB Allocation?
# Calculate the average number of PRBs used per BS for SAC and SAC_pred
# avg_prbs_sac = mat_used_prbs_per_user_per_bs.mean(axis=(0,2,3))
# # avg_prbs_sac_pred = mat_used_prbs_per_user_per_bs_pred.mean(axis=(0,2,3))

# # Create an array with the positions of each bar on the x-axis
# barWidth = 0.3
# r1 = np.arange(len(avg_prbs_sac))
# r2 = [x + barWidth for x in r1]

# # Create the bar chart
# plt.bar(r1, avg_prbs_sac, color='b', width=barWidth, edgecolor='grey', label='SAC')
# # plt.bar(r2, avg_prbs_sac_pred, color='r', width=barWidth, edgecolor='grey', label='SAC_pred')

# # Add xticks on the middle of the group bars
# plt.xlabel('BS', fontweight='bold')
# plt.xticks([r + barWidth/2 for r in range(BS_NO)], range(1, BS_NO+1))

# plt.ylabel('Average number of PRBs used')
# plt.legend()

# # Show the plot
# plt.show()

########################
# Select data for BS=0
prbs_per_user_per_bs_0 = mat_used_prbs_per_user_per_bs[:, 4, :, :]
# prbs_per_user_per_bs_pred_0 = mat_used_prbs_per_user_per_bs_pred[:, 4, :, :]

# Sum over users and then calculate averages over Monte Carlo runs
sum_prbs_per_user_per_bs_0 = np.sum(prbs_per_user_per_bs_0, axis=1)
# sum_prbs_per_user_per_bs_pred_0 = np.sum(prbs_per_user_per_bs_pred_0, axis=1)

avg_prbs_per_user_per_bs_0 = np.mean(sum_prbs_per_user_per_bs_0, axis=0)
# avg_prbs_per_user_per_bs_pred_0 = np.mean(sum_prbs_per_user_per_bs_pred_0, axis=0)

# Plot the averages using your function
plot_graph('Overall PRBs used for BS=4 for SAC algorithm',
           [avg_prbs_per_user_per_bs_0],
           ['SAC'],
           ['b'],
           ['-'],
           'T',
           'Overall PRBs used in BS 4')
# plot_graph('Overall PRBs used for BS=4 for SAC and SAC_pred algorithms',
#            [avg_prbs_per_user_per_bs_0, avg_prbs_per_user_per_bs_pred_0],
#            ['SAC', 'SAC_pred'],
#            ['b', 'r'],
#            ['-', '--'],
#            'T',
#            'Overall PRBs used in BS 4')
#######################
# Calculate averages over Monte Carlo runs and users
avg_prbs_per_user = np.mean(mat_used_prbs_per_user, axis=(0,1))
# avg_prbs_per_user_pred = np.mean(mat_used_prbs_per_user_pred, axis=(0,1))

# Plot the averages using your function
plot_graph('Avg PRBs used per user for SAC algorithm',
           [avg_prbs_per_user],
           ['SAC'],
           ['b'],
           ['-'],
           'T',
           'Average PRBs used per user')
# plot_graph('Avg PRBs used per user for SAC and SAC_pred algorithms',
#            [avg_prbs_per_user, avg_prbs_per_user_pred],
#            ['SAC', 'SAC_pred'],
#            ['b', 'r'],
#            ['-', '--'],
#            'T',
#            'Average PRBs used per user')
# %%REWARD%%%%
mat_reward_average_over_m = np.average(mat_reward, axis=0)
# mat_reward_average_over_m_pred = np.average(mat_reward_pred, axis=0)

mean_ep_rewardall = moving_average(mat_reward_average_over_m, window_size)
# mean_ep_rewardall_pred = moving_average(mat_reward_average_over_m_pred, window_size)
plot_graph("Mean episodic rewards",
           [mean_ep_rewardall],
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
mean_ep_prb_const = moving_average(np.average(mat_satisfied_prb_constraint, axis=0), window_size)
# mean_ep_prb_const_pred = moving_average(np.average(mat_satisfied_prb_constraint_pred, axis=0), window_size)
mean_ep_power_const = moving_average(np.average(mat_satisfied_power_constraint, axis=0), window_size)
# mean_ep_power_const_pred = moving_average(np.average(mat_satisfied_power_constraint_pred, axis=0), window_size)
mean_ep_delay_const = moving_average(np.average(mat_satisfied_delay_constraint, axis=0), window_size)
# mean_ep_delay_const_pred = moving_average(np.average(mat_satisfied_delay_constraint_pred, axis=0), window_size)

plot_graph("Constraint Satisfaction",
           [mean_ep_prb_const, mean_ep_power_const, mean_ep_delay_const],
           ['PRB (SAC)',
            'Power (SAC)',
            'Delay (SAC)'],
           ['red', 'blue', 'green'],
           ['solid', 'solid', 'solid'],
           "Episode",
           "Constraint Satisfaction Rate")

# plot_graph("Constraint Satisfaction",
#            [mean_ep_prb_const, mean_ep_prb_const_pred,
#             mean_ep_power_const, mean_ep_power_const_pred,
#             mean_ep_delay_const, mean_ep_delay_const_pred],
#            ['PRB (SAC)', 'PRB (SAC_pred)',
#             'Power (SAC)', 'Power (SAC_pred)',
#             'Delay (SAC)', 'Delay (SAC_pred)'],
#            ['red', 'red', 'blue', 'blue', 'green', 'green'],
#            ['solid', 'dotted', 'solid', 'dotted', 'solid', 'dotted'],
#            "Episode",
#            "Constraint Satisfaction")
# %%%%%SSL%%%%%%
mean_ep_ssl_rate = moving_average(np.average(mat_ssl_rate, axis=0), window_size)
mean_ep_ssl_delay = moving_average(np.average(mat_ssl_delay, axis=0), window_size)
mean_ep_ssl = moving_average(np.average(mat_ssl, axis=0), window_size)
# mean_ep_ssl_rate_pred = moving_average(np.average(mat_ssl_rate_pred, axis=0), window_size)
# mean_ep_ssl_delay_pred = moving_average(np.average(mat_ssl_delay_pred, axis=0), window_size)
# mean_ep_ssl_pred = moving_average(np.average(mat_ssl_pred, axis=0), window_size)

# plot_graph("SSL Metrics",
#            [mean_ep_ssl_rate, mean_ep_ssl_rate_pred,
#             mean_ep_ssl_delay, mean_ep_ssl_delay_pred,
#             mean_ep_ssl, mean_ep_ssl_pred],
#            ['Rate (SAC)', 'Rate (SAC_pred)',
#             'Delay (SAC)', 'Delay (SAC_pred)',
#             'SSL (SAC)', 'SSL (SAC_pred)'],
#            ['blue', 'blue', 'green', 'green', 'orange', 'orange'],
#            ['solid', 'dotted', 'solid', 'dotted', 'solid', 'dotted'],
#            "Episode",
#            "SSL Metrics")
plot_graph("SSL Metrics",
           [mean_ep_ssl_rate,
            mean_ep_ssl_delay,
            mean_ep_ssl],
           ['Rate (SAC)',
            'Delay (SAC)',
            'SSL (SAC)'],
           ['blue', 'green', 'orange'],
           ['solid', 'solid', 'solid'],
           "Episode",
           "SSL Metrics")
# %%%%%Delay%%%%%%
# Calculate mean delay over users for SAC
mean_delay_sac = np.mean(np.mean(monte_mat_delay_tot, axis=1), axis=0)
# Calculate mean delay over users for SAC_pred
# mean_delay_sac_pred = np.mean(np.mean(monte_mat_delay_tot_pred, axis=1), axis=0)

# Apply moving average to smooth the curves
mean_delay_sac_smoothed = moving_average(mean_delay_sac, window_size)
# mean_delay_sac_pred_smoothed = moving_average(mean_delay_sac_pred, window_size)

# Plot the comparison graph
plot_graph("Comparison of Average E2E Delay (SAC)",
           [mean_delay_sac_smoothed],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Timestep",
           "Average E2E Delay (ms)")
# plot_graph("Comparison of Average E2E Delay (SAC vs. SAC_pred)",
#            [mean_delay_sac_smoothed, mean_delay_sac_pred_smoothed],
#            ['SAC', 'SAC_pred'],
#            ['blue', 'green'],
#            ['solid', 'solid'],
#            "Timestep",
#            "Average E2E Delay (ms)")
#####################
mean_rate_sac = np.mean(np.mean(shannon, axis=1), axis=0)
mean_rate_sac_smoothed = moving_average(mean_rate_sac, window_size)
plot_graph("Average Data Rate (SAC)",
           [mean_rate_sac_smoothed],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Timestep",
           "Average Data Rate (Mbps)")

#####################
mean_power_sac = np.mean(np.mean(mat_power, axis=1), axis=0)
mean_power_sac_smoothed = moving_average(mean_power_sac, window_size)
plot_graph("Average of total allocated power to each user (SAC)",
           [mean_power_sac_smoothed],
           ['SAC'],
           ['blue'],
           ['solid'],
           "Timestep",
           "Average of total allocated power to each user (W)")

print(style.UNDERLINE + "Total time for {} timeslots/episodes ({} users) in {} Monte-Carlo iterations: {}".format(T, USER_NO, MC, convert_seconds(np.sum(mat_episode_runtime))))

