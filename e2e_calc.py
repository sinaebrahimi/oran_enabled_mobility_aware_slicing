import numpy as np
from radio_calc import Location


class Mapping:
    #def __init__(self, action, FH_BW_CAPACITY, E2_BW_CAPACITY, du_ru_adj_matrix, ric_du_adj_matrix, mat_fh_links_capacity, mat_e2_links_capacity, , mat_specs, associator, USER_NO, BS_NO, DU_NO, PRB_NO, MAX_POWER):
    def __init__(self, action, mat_specs, associator, USER_NO, BS_NO, PRB_NO, MAX_POWER):
        self.USER_NO = USER_NO
        self.BS_NO = BS_NO
        #self.DU_NO = DU_NO
        self.PRB_NO = PRB_NO
        self.MAX_POWER = MAX_POWER
        #---------
        self.action = action
        # --------------------------------------------------------------------------
        self.P = np.zeros([self.BS_NO, self.PRB_NO, self.USER_NO])
        self.rho = np.zeros([self.BS_NO, self.PRB_NO, self.USER_NO])
        #############################################################
        self.remained_power = np.ones([self.BS_NO]) * self.MAX_POWER
        #############################################################
        # self.FH_BW_CAPACITY = FH_BW_CAPACITY
        # self.E2_BW_CAPACITY = E2_BW_CAPACITY
        # self.mat_fh_links_capacity = mat_fh_links_capacity
        # self.mat_e2_links_capacity = mat_e2_links_capacity
        # self.du_ru_adj_matrix = du_ru_adj_matrix
        # self.ric_du_adj_matrix = ric_du_adj_matrix
        self.mat_specs = mat_specs
        #############################################################
        # self.temp_mat_fh_links_capacity = np.copy(mat_fh_links_capacity)
        # self.temp_mat_e2_links_capacity = np.copy(mat_e2_links_capacity)
        #############################################################
        self.associator = associator
#%% RAN mapping:
    # def fh_e2_remaining_capacity(self):    #SKIPPING FOR NOW    
    #     return self.temp_mat_fh_links_capacity, self.temp_mat_e2_links_capacity
        
    def ran_prb_allocation(self): # Equivalent to \rho^{b}_{o,u}(t) in the paper; PRB allocation
        self.e0 = 0
        self.e1 = self.BS_NO * self.PRB_NO * self.USER_NO
        self.temp_rho = self.action[self.e0:self.e1]
        self.temp_rho_reshaped = np.reshape(self.temp_rho, [self.BS_NO, self.PRB_NO, self.USER_NO])
        self.done_user_prb_allocation = 0
        cnt_u = 0
        for u in range(self.USER_NO):
            for b in range(self.BS_NO):
                if self.associator[u, b] == 1:
                    for k in range(self.PRB_NO):
                        if np.sum(self.rho[b, k, :]) == 0:###########
                            if self.temp_rho_reshaped[b, k, u] > .5: # rounding up
                                self.rho[b, k, u] = 1

            if np.sum(self.rho[:, :, u]) > 0:
                cnt_u += 1

        if cnt_u == self.USER_NO:
            self.done_user_prb_allocation = 1

        return self.done_user_prb_allocation, self.rho
    

    def ran_prb_allocation_greedy_version(self):
        self.e0 = 0
        self.e1 = self.BS_NO * self.PRB_NO * self.USER_NO
        self.temp_rho = self.action[self.e0:self.e1]
        self.temp_rho_reshaped = np.reshape(self.temp_rho, [self.BS_NO, self.PRB_NO, self.USER_NO])
        # Initialize variables
        self.done_user_prb_allocation = 0
        cnt_u=0

        # Greedy allocation
        for u in range(self.USER_NO):
            for b in range(self.BS_NO):
                if self.associator[u, b] == 1:
                    for k in range(self.PRB_NO):
                        if self.temp_rho_reshaped[b, k, u] > 0: 
                        # if self.temp_rho_reshaped[b, k, u] > 0.5: it is between -1 and 1, so i should change it to 0... 
                            self.rho[b, k, u] = 1
            if np.sum(self.rho[:, :, u]) > 0:
                cnt_u += 1

        if cnt_u == self.USER_NO:
            self.done_user_prb_allocation = 1

        return self.done_user_prb_allocation, self.rho
    

    def ran_power_allocation(self):
        self.e2 = self.e1 + self.BS_NO * self.PRB_NO * self.USER_NO
        self.temp_p = (self.action[self.e1:self.e2]+1)/2 ### why?
        self.temp_p_reshaped = np.reshape(self.temp_p, [self.BS_NO, self.PRB_NO, self.USER_NO])
        self.scale = self.MAX_POWER / self.PRB_NO #0.1 # What is it? #making it 1 to make it ineffective #it was 0.01 first
        # self.scale = .01
        self.done_user_power_allocation = 0
        cnt_u = 0
        for u in range(self.USER_NO):
            for b in range(self.BS_NO):
                if self.associator[u, b] == 1:
                    for k in range(self.PRB_NO):
                        if (self.rho[b, k, u]) == 1:
                            if self.remained_power[b] - (self.scale * self.temp_p_reshaped[b, k, u]) > 0:
                                self.P[b, k, u] = self.scale * self.temp_p_reshaped[b, k, u]
                                self.remained_power[b] -= self.P[b, k, u]

            if np.sum(self.P[:, :, u]) > 0:
                cnt_u += 1
        if cnt_u == self.USER_NO:
            self.done_user_power_allocation = 1

        return self.done_user_power_allocation, self.P



#%%%%
class Delay:
    def __init__(self, mat_rate, FH_BW_CAPACITY, E2_BW_CAPACITY, mat_specs, associator, mat_distance_uu, distances_ric_du, distances_du_ru, du_ru_adj_matrix, ric_du_adj_matrix, USER_NO, BS_NO, DU_NO):
        self.USER_NO = USER_NO
        self.BS_NO = BS_NO
        self.DU_NO = DU_NO
        #-------------------
        self.mat_rate = mat_rate
        self.FH_BW_CAPACITY = FH_BW_CAPACITY
        self.associator = associator
        self.E2_BW_CAPACITY = E2_BW_CAPACITY
        self.mat_specs = mat_specs
        self.speed_of_light = 3e8 # 300000000 # 3*10^8 m/s (exactly equal to 299,792,458 metres per second)
        self.mat_distance_uu = mat_distance_uu
        self.distances_ric_du = distances_ric_du
        self.distances_du_ru = distances_du_ru
        self.du_ru_adj_matrix = du_ru_adj_matrix
        self.ric_du_adj_matrix = ric_du_adj_matrix

        self.mat_delay_prop_uu = np.zeros([self.USER_NO])
        self.mat_delay_prop_e2 = np.zeros([self.USER_NO])
        self.mat_delay_prop_fh = np.zeros([self.USER_NO])

        self.mat_delay_tx_uu = np.zeros([self.USER_NO])
        self.mat_delay_tx_fh = np.zeros([self.USER_NO])
        self.mat_delay_tx_e2 = np.zeros([self.USER_NO])
#%%%%%%%%% Propagation delay calculation:
    def prop_fh_e2(self):
        for u in range(self.USER_NO):
            for du in range(self.DU_NO):
                for b in range(self.BS_NO):
                    if self.du_ru_adj_matrix[du, b] == 1: # instead of \zeta ζ in the paper                       
                        if self.associator[u, b] == 1:
                            distance_du_ru = self.distances_du_ru[du, b]
                            self.mat_delay_prop_fh[u] = distance_du_ru / self.speed_of_light
                            
                            distance_ric_du = self.distances_ric_du[0, du]
                            self.mat_delay_prop_e2[u] = distance_ric_du / self.speed_of_light
                            
        return self.mat_delay_prop_fh, self.mat_delay_prop_e2
    
    def prop_uu(self):
        for u in range(self.USER_NO):
            for b in range(self.BS_NO):
                if self.associator[u, b] == 1: # instead of multiplying \chi
                   distance = self.mat_distance_uu[b, u]
                   self.mat_delay_prop_uu[u] = distance / self.speed_of_light
        return self.mat_delay_prop_uu

#%%%%%%%%% Transmission delay calculation:
    def tx_fh_e2(self):        
        for u in range(self.USER_NO):
            PACKET_SIZE = self.mat_specs[u, 3] # in mbits
            # PACKET_NO = self.mat_specs[u, 4] # number of packetsper each second
            fh_overhead = 0.1 * PACKET_SIZE
            e2_overhead = 0.1 * PACKET_SIZE
            for du in range(self.DU_NO):
                for b in range(self.BS_NO):
                    if self.du_ru_adj_matrix[du, b] == 1: # instead of \zeta ζ in the paper                       
                        if self.associator[u, b] == 1:
                            if self.mat_rate[u] == 0:
                                delay_tx_fh = 0
                                delay_tx_e2 = 0
                            else:
                                delay_tx_fh = (PACKET_SIZE + fh_overhead) / self.FH_BW_CAPACITY
                                delay_tx_e2 = (PACKET_SIZE + fh_overhead + e2_overhead) / self.E2_BW_CAPACITY
                            self.mat_delay_tx_fh[u] = delay_tx_fh
                            self.mat_delay_tx_e2[u] = delay_tx_e2                            
        return self.mat_delay_tx_fh, self.mat_delay_tx_e2
    
    def tx_uu(self):
        for u in range(self.USER_NO):
            PACKET_SIZE = self.mat_specs[u, 3] # in mbits
            #PACKET_NO = self.mat_specs[u, 4] # number of packetsper each second
            if self.mat_rate[u] == 0:
                delay_tx_uu = 0
            else:
                delay_tx_uu = (PACKET_SIZE) / self.mat_rate[u]
                # delay_tx_uu = (PACKET_SIZE * PACKET_NO) / self.mat_rate[u]
            self.mat_delay_tx_uu[u] = delay_tx_uu
        return self.mat_delay_tx_uu


    def _(self):
        self.mat_delay_tot = np.ones([self.USER_NO], dtype=np.float64) # avoiding zeros to escape division to zero error # to multiply to something smaller as we will multiply it by 1000 to convert to ms
        self.mat_delay_tot *= 0.00001 # 10 us (0.01 ms)
        done_delay_all = 0
        cnt_u = 0
        self.mat_delay_tx_uu = Delay.tx_uu(self)
        self.mat_delay_tx_fh, self.mat_delay_tx_e2 = Delay.tx_fh_e2(self)
        self.mat_delay_prop_fh, self.mat_delay_prop_e2 = Delay.prop_fh_e2(self)
        self.mat_delay_prop_uu = Delay.prop_uu(self)

        for u in range(self.USER_NO):
            user_tolerable_delay = self.mat_specs[u, 2] / 1000 # because it was in ms
            # if self.mat_delay_tx_cn[u] + self.mat_delay_proc[u] + self.mat_delay_proc[u] < user_tolerable_delay:
            total_delay_user = np.float64(self.mat_delay_tx_uu[u] + self.mat_delay_tx_fh[u] + self.mat_delay_tx_e2[u] + self.mat_delay_prop_uu[u] + self.mat_delay_prop_e2[u] + self.mat_delay_prop_fh[u]) # check self.mat_delay_tx_uu
            self.mat_delay_tot[u] = total_delay_user
            if self.mat_delay_tot[u] < user_tolerable_delay:
                cnt_u += 1                
                #print("User {} total delay: {}".format(u, self.mat_delay_tot[u]))###to remove

        if cnt_u == self.USER_NO:
            done_delay_all = 1

        self.mat_delay_tot = self.mat_delay_tot * 1000   #  To convert to ms
        return cnt_u, done_delay_all, self.mat_delay_tot

# %%
class StateCalculation: # TO BE COMPLETED
    def __init__(self, loc_users_t, H, X_LIM):
        self.loc_users_t = loc_users_t
        self.H = H
        self.X_LIM = X_LIM
        #to add other states
        # self.mat_specs = mat_specs # s3 = 4 * USER_NO # includes 0: selected slice, 1: min rate, 2: tolerable delay, 3: packet size

    def _(self):
        self.states_no = self.loc_users_t.size + self.H.size 
        self.state = np.zeros([self.states_no])

        self.loc_users_reshaped = np.reshape(self.loc_users_t, [self.loc_users_t.size])
        self.H_reshaped = np.reshape(self.H, [self.H.size])


        self.state[0:self.loc_users_t.size] = self.loc_users_reshaped / self.X_LIM #Normalizing to /1000?
        self.state[self.loc_users_t.size:self.loc_users_t.size + self.H.size] = 100 * (self.H_reshaped)/(np.max(self.H_reshaped)) # Normalizing the value for the neural network (avoiding errors)

        # cant understand multiplications to 100

        # self.state[0:self.H.size] = 100 * (self.H_reshaped)/(np.max(self.H_reshaped)) # Normalizing the value for the neural network (avoiding errors)
        # self.state[self.H.size:self.H.size + self.loc_users_t.size] = self.loc_users_reshaped #SHOULDN't WE ALSO NORMALIZE THIS TO /1000?
        self.state = 100 * self.state / np.max(self.state) #  the last normalization!
        return self.state

