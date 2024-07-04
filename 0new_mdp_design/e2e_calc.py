import numpy as np
from radio_calc import Location


class Mapping:
    #def __init__(self, action, FH_BW_CAPACITY, E2_BW_CAPACITY, du_ru_adj_matrix, ric_du_adj_matrix, mat_fh_links_capacity, mat_e2_links_capacity, , mat_specs, associator, USER_NO, BS_NO, DU_NO, PRB_NO, MAX_POWER):
    def __init__(self, action, mat_specs, H_b, USER_NO, BS_NO, PRB_NO, MAX_POWER):
    # def __init__(self, action, mat_specs, associator, USER_NO, BS_NO, PRB_NO, MAX_POWER):
        self.USER_NO = USER_NO
        self.BS_NO = BS_NO
        #self.DU_NO = DU_NO
        self.PRB_NO = PRB_NO
        self.MAX_POWER = MAX_POWER
        #---------
        self.action = action
        # --------------------
        self.H_b = H_b
        # ------------------------------------------------------
        self.chi = np.zeros([self.USER_NO, self.BS_NO]) # user-bs associator
        self.chi_num = np.zeros([self.USER_NO])
        self.p = np.zeros([self.BS_NO, self.PRB_NO, self.USER_NO])
        self.p_num = np.zeros([self.USER_NO])
        self.rho = np.zeros([self.BS_NO, self.PRB_NO, self.USER_NO])
        self.rho_num = np.zeros([self.USER_NO])
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
#%% RAN mapping:
    def user_association(self, chi_num_prev): # chi_num_prev[u]
        self.e0 = 0
        self.e1 = self.USER_NO
        self.temp_chi = (self.action[self.e0:self.e1])
        self.temp_chi = np.clip(self.temp_chi, 0, 1) # to avoid negative values # to make sure that the values are between 0 and 1
        self.temp_chi_reshaped = np.reshape(self.temp_chi, [self.USER_NO])

        self.chi_action = self.temp_chi_reshaped

        for u in range(self.USER_NO):
            user_channel_gains = self.H_b[:, u]
            sorted_indices = np.argsort(-user_channel_gains)  # Sort in descending order
            selected_bs_prev = int(chi_num_prev[u]) # prefer the old assignment
            first_bs_now = int(sorted_indices[0]) # prefer the first best BS in current time step
            second_bs_now = int(sorted_indices[1]) # prefer the second best BS in current time step

            if self.chi_action[u] <= 0.33: # prefer the old assignment
                b = selected_bs_prev # int(chi_num_prev[u])
            elif self.chi_action[u] > 0.66: # prefer the second best BS in current time step
                b = second_bs_now
            else: # prefer the first best BS in current time step
                b = first_bs_now

            # Assign the chosen BS to the user
            self.chi[u, b] = 1
            # Update chi_num with the chosen BS index
            self.chi_num[u] = int(b)

        return self.chi_num, self.chi, self.chi_action
    
    def user_association_t0(self):
        # argmax(H_b)
        self.chi_action = np.zeros([self.USER_NO]) # means that all users stayed in their BS from the t=-1 to t=0!
        user_channel_gains = self.H_b.T
        self.chi_num = np.argmax(user_channel_gains, axis=1) # select the highest H_b for each user
        self.chi[np.arange(self.USER_NO), self.chi_num] = 1

        return self.chi_num, self.chi, self.chi_action


    def ran_prb_allocation(self): # Equivalent to \rho^{r_d}_{k,u}(t) in the paper; PRB allocation
        self.e2 = self.e1 +  self.USER_NO * self.PRB_NO
        self.temp_rho = (self.action[self.e1:self.e2])
        self.temp_rho_reshaped = np.reshape(self.temp_rho, [self.USER_NO, self.PRB_NO])
        self.rho_action = self.temp_rho_reshaped
        self.done_user_prb_allocation = 0
        cnt_u = 0 

        for u in range(self.USER_NO):
            for b in range(self.BS_NO):
                if self.chi_num[u] == b: #self.chi_num[u,b] == 1:
                    for k in range(self.PRB_NO):
                        if np.sum(self.rho[b, k, :]) == 0: # is the PRB allocated to another user or not?
                            if self.rho_action[u, k] >= 0.5: # binarizing the action for rho
                                self.rho[b, k, u] = 1
            if np.sum(self.rho[:, :, u]) > 0:
                cnt_u += 1

        if cnt_u == self.USER_NO:
            self.done_user_prb_allocation = 1

        for u in range(self.USER_NO):
            bs = int(self.chi_num[u])  # Get the BS to which the user is connected
            self.rho_num[u] = int(np.sum(self.rho[bs, :, u]))  # Sum the PRBs assigned to this user
        return  self.rho_num, self.rho, self.rho_action

    def ran_prb_allocation_t0(self): #chi_num, mat_specs, BS_NO, PRB_NO, USER_NO
        self.rho_action = np.zeros([self.USER_NO, self.PRB_NO]) # we already know user-BS assignment from chi

        # Initialize PRB allocation matrix
        prb_allocation = np.zeros((self.BS_NO, self.USER_NO))

        for b in range(self.BS_NO):
            connected_users = [u for u in range(self.USER_NO) if self.chi_num[u] == b]
            if not connected_users:
                continue

            # Calculate weights based on rate requirements and channel gains
            weights = np.zeros(len(connected_users))
            for i, u in enumerate(connected_users):
                rate_requirement = self.mat_specs[u, 1]
                weights[i] = rate_requirement

            # Normalize weights
            max_weight = np.max(weights)
            if max_weight > 0:
                weights = weights / max_weight

            # Apply softmax to weights to get allocation proportions
            exp_weights = np.exp(weights)  # Removed subtraction of max(weights) for stability
            allocation_proportions = exp_weights / np.sum(exp_weights)

            # Allocate PRBs based on allocation proportions
            allocated_prbs = (allocation_proportions * self.PRB_NO).astype(int)
            
            # Ensure the sum of allocated PRBs does not exceed PRB_NO
            while np.sum(allocated_prbs) < self.PRB_NO:
                allocated_prbs[np.argmax(allocation_proportions)] += 1
            while np.sum(allocated_prbs) > self.PRB_NO:
                allocated_prbs[np.argmax(allocated_prbs)] -= 1

            # Fill the prb_allocation and rho matrices sequentially
            next_prb_index = 0
            for i, u in enumerate(connected_users):
                prb_allocation[b, u] = allocated_prbs[i]
                self.rho[b, next_prb_index:next_prb_index + allocated_prbs[i], u] = 1
                next_prb_index += allocated_prbs[i]

        for u in range(self.USER_NO):
            bs = self.chi_num[u]  # Get the BS to which the user is connected
            self.rho_num[u] = int(np.sum(self.rho[bs, :, u]))  # Sum the PRBs assigned to this user

            # self.rho_action[u, :] = self.rho[bs, :, u]  
            for k in range(self.PRB_NO):
                if self.rho[bs, k, u] == 0:
                    self.rho_action[u, k] = np.random.uniform(0.01, 0.49)
                else:
                    self.rho_action[u, k] = np.random.uniform(0.51, 0.99) # mimicking the interpretation we expect for rho_action

        return self.rho_num, self.rho, self.rho_action

    
    def ran_power_allocation(self):
        self.e3 = self.e2 + self.USER_NO * self.PRB_NO
        self.temp_p = (self.action[self.e2:self.e3])
        self.temp_p = np.clip(self.temp_p, 0.1, 1) # to avoid 0 values for power
        self.temp_p_reshaped = np.reshape(self.temp_p, [self.USER_NO, self.PRB_NO])
        self.p_action = self.temp_p_reshaped

        self.power_scale = self.MAX_POWER / self.PRB_NO
        self.done_user_power_allocation = 0
        cnt_u = 0

        for u in range(self.USER_NO):
            for b in range(self.BS_NO):
                if self.chi_num[u] == b: #self.chi_num[u,b] == 1:
                    for k in range(self.PRB_NO):
                        if self.rho[b, k, u] == 1:
                            #if self.remained_power[b] - (self.power_scale * self.temp_p_reshaped[b, k, u]) > 0:
                            self.p[b, k, u] = self.power_scale * self.p_action[u, k]
                            # self.remained_power[b] -= self.p[b, k, u]

            if np.sum(self.p[:, :, u]) > 0:
                cnt_u += 1

        if cnt_u == self.USER_NO:
            self.done_user_power_allocation = 1

        for u in range(self.USER_NO):
            bs = int(self.chi_num[u])  # Get the BS to which the user is connected
            self.p_num[u] = np.sum(self.p[bs, :, u]) 

        return self.p_num, self.p, self.p_action    
    
    def ran_power_allocation_t0(self): 
        self.p_action = np.zeros([self.USER_NO, self.PRB_NO]) # we already know user-BS assignment from chi, so no need to include BS_NO
        self.power_scale = self.MAX_POWER / self.PRB_NO
        for b in range(self.BS_NO):
            users_connected = np.where(self.chi_num == b)[0]
            if len(users_connected) > 0:
                for k in range(self.PRB_NO):
                    for i, u in enumerate(users_connected):
                        if self.rho[b, k, u] == 1:
                            self.p[b, k, u] = self.power_scale
                            self.p_action[u, k] = 1 # self.power_scale # maybe it's not good

        for u in range(self.USER_NO):
            bs = self.chi_num[u]  # Get the BS to which the user is connected
            self.p_num[u] = np.sum(self.p[bs, :, u]) 

        return self.p_num, self.p, self.p_action

##############
class Delay:
    def __init__(self, mat_rate, FH_BW_CAPACITY, E2_BW_CAPACITY, mat_specs, chi, mat_distance_uu, distances_ric_du, distances_du_ru, du_ru_adj_matrix, ric_du_adj_matrix, USER_NO, BS_NO, DU_NO):
        self.USER_NO = USER_NO
        self.BS_NO = BS_NO
        self.DU_NO = DU_NO
        #-------------------
        self.mat_rate = mat_rate
        self.FH_BW_CAPACITY = FH_BW_CAPACITY
        self.chi = chi
        self.E2_BW_CAPACITY = E2_BW_CAPACITY
        self.mat_specs = mat_specs
        self.speed_of_light = 3e8 # 300000000 # 3*10^8 m/s (exactly eq5ual to 299,792,458 metres per second)
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
                        if self.chi[u, b] == 1:
                            distance_du_ru = self.distances_du_ru[du, b]
                            self.mat_delay_prop_fh[u] = distance_du_ru / self.speed_of_light
                            
                            distance_ric_du = self.distances_ric_du[0, du]
                            self.mat_delay_prop_e2[u] = distance_ric_du / self.speed_of_light
                            
        return self.mat_delay_prop_fh, self.mat_delay_prop_e2
    
    def prop_uu(self):
        for u in range(self.USER_NO):
            for b in range(self.BS_NO):
                if self.chi[u, b] == 1: # instead of multiplying \chi
                   distance = self.mat_distance_uu[b, u]
                   self.mat_delay_prop_uu[u] = distance / self.speed_of_light
        return self.mat_delay_prop_uu

#%%%%%%%%% Transmission delay calculation:
    def tx_fh_e2(self):
        # fronthaul links used BW:
        fh_used_bw = np.zeros([self.DU_NO, self.BS_NO])
        is_fh_capacity_full = np.zeros([self.DU_NO, self.BS_NO]) # # Initialize violation flags array with False values
        e2_used_bw = np.zeros([self.DU_NO])
        is_e2_capacity_full = np.zeros([self.DU_NO])
        for u in range(self.USER_NO):
            PACKET_SIZE = self.mat_specs[u, 3] # in mbits
            # PACKET_NO = self.mat_specs[u, 4] # number of packetsper each second
            fh_overhead = 0.1 * PACKET_SIZE
            e2_overhead = 0.1 * PACKET_SIZE
            for du in range(self.DU_NO):
                for b in range(self.BS_NO):
                    if self.du_ru_adj_matrix[du, b] == 1: # instead of \zeta ζ in the paper                       
                        if self.chi[u, b] == 1:
                            # count how much of the link is being used by summing up rates
                            # then check if fh_link_used_bw exceeds self.FH_BW_CAPACITY
                            fh_used_bw[du, b] += self.mat_rate[u]
                            e2_used_bw[du] += self.mat_rate[u]
                            if fh_used_bw[du, b] >= self.FH_BW_CAPACITY:
                                is_fh_capacity_full[du, b] = 1

                            if e2_used_bw[du] >= self.E2_BW_CAPACITY:
                                is_e2_capacity_full[du] = 1


                            ###delay calculation
                            if self.mat_rate[u] == 0:
                                delay_tx_fh = 0
                                delay_tx_e2 = 0
                            else:
                                delay_tx_fh = (PACKET_SIZE + fh_overhead) / self.FH_BW_CAPACITY
                                delay_tx_e2 = (PACKET_SIZE + fh_overhead + e2_overhead) / self.E2_BW_CAPACITY
                            self.mat_delay_tx_fh[u] = delay_tx_fh
                            self.mat_delay_tx_e2[u] = delay_tx_e2                            
        return self.mat_delay_tx_fh, self.mat_delay_tx_e2, is_fh_capacity_full, is_e2_capacity_full
    
    def tx_uu(self):
        flag_uu_failure_due_to_rate = np.zeros([self.USER_NO])
        for u in range(self.USER_NO):
            PACKET_SIZE = self.mat_specs[u, 3] # in mbits
            #PACKET_NO = self.mat_specs[u, 4] # number of packetsper each second
            if self.mat_rate[u] == 0:
                delay_tx_uu = 0
                flag_uu_failure_due_to_rate[u] = 1
            else:
                if self.mat_rate[u] < PACKET_SIZE:
                    delay_tx_uu = 0
                    flag_uu_failure_due_to_rate[u] = 1
                else:
                    delay_tx_uu = (PACKET_SIZE) / self.mat_rate[u]
                # delay_tx_uu = (PACKET_SIZE * PACKET_NO) / self.mat_rate[u]
            self.mat_delay_tx_uu[u] = delay_tx_uu
        return self.mat_delay_tx_uu, flag_uu_failure_due_to_rate


    def _(self):
        self.mat_delay_tot = np.ones([self.USER_NO], dtype=np.float64) # avoiding zeros to escape division to zero error # to multiply to something smaller as we will multiply it by 1000 to convert to ms
        self.mat_delay_tot *= 0.00001 # 10 us (0.01 ms)
        done_delay_all = 0
        cnt_u = 0
        self.mat_delay_tx_uu, flag_uu_failure_due_to_rate = Delay.tx_uu(self)
        self.mat_delay_tx_fh, self.mat_delay_tx_e2, is_fh_capacity_full, is_e2_capacity_full = Delay.tx_fh_e2(self)
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
        return cnt_u, done_delay_all, self.mat_delay_tot, is_fh_capacity_full, is_e2_capacity_full, flag_uu_failure_due_to_rate

# %%
class StateCalculation: # TO BE COMPLETED
    def __init__(self, mat_ssl_u_total, prb_util, prb_ratio, power_util, power_ratio): # all from time step (t-1)
        #self, H, mat_ssl_u_total, mat_chi, mat_rho, mat_p
        # self.H = H # USER_NO * BS_NO
        self.mat_ssl_u_total = mat_ssl_u_total # USER_NO
        self.prb_util = prb_util # BS_NO
        self.prb_ratio = prb_ratio  # USER_NO
        self.power_util = power_util # BS_NO
        self.power_ratio = power_ratio  # USER_NO
        #to add other states
        # self.mat_specs = mat_specs # s3 = 4 * USER_NO # includes 0: selected slice, 1: min rate, 2: tolerable delay, 3: packet size

    def _(self):
        # self.s1 = self.H.size # b*u
        self.s1 = self.mat_ssl_u_total.size # USER_NO
        self.s2 = self.s1 + self.prb_util.size # BS_NO
        self.s3 = self.s2 + self.prb_ratio.size # USER_NO
        self.s4 = self.s3 + self.power_util.size # BS_NO
        self.s5 = self.s4 + self.power_ratio.size # USER_NO
        self.states_no = self.s5
        #self.H.size + self.mat_ssl_u_rate.size + self.mat_ssl_u_delay.size + self.mat_chi_compressed.size + self.mat_rho_compressed.size + self.mat_p_compressed.size
        self.state = np.zeros([self.states_no])

        # self.loc_users_reshaped = np.reshape(self.loc_users_t, [self.loc_users_t.size])
        # self.H_reshaped = np.reshape(self.H, [self.H.size])
        self.mat_ssl_u_total_reshaped = np.reshape(self.mat_ssl_u_total, [self.mat_ssl_u_total.size])
        self.prb_util_reshaped = np.reshape(self.prb_util, [self.prb_util.size])
        self.prb_ratio_reshaped = np.reshape(self.prb_ratio, [self.prb_ratio.size])
        self.power_util_reshaped = np.reshape(self.power_util, [self.power_util.size])
        self.power_ratio_reshaped = np.reshape(self.power_ratio, [self.power_ratio.size])

        # H
        # self.state[0:self.s1] = (self.H_reshaped)/(np.max(self.H_reshaped)) # Normalizing the value for the neural network (avoiding errors)
        
        # mat_ssl_u_total
        self.state[0:self.s1] = self.mat_ssl_u_total_reshaped

        self.state[self.s1:self.s2] = self.prb_util_reshaped      
        self.state[self.s2:self.s3] = self.prb_ratio_reshaped #  no need to normalize
        self.state[self.s3:self.s4] = self.power_util_reshaped #  no need to normalize
        self.state[self.s4:self.s5] = self.power_ratio_reshaped #  no need to normalize

        # self.state[0:self.loc_users_t.size] = self.loc_users_reshaped / self.X_LIM #Normalizing to /1000?
        # self.state[self.loc_users_t.size:self.loc_users_t.size + self.H.size] = 100 * (self.H_reshaped)/(np.max(self.H_reshaped)) # Normalizing the value for the neural network (avoiding errors)

        # cant understand multiplications to 100

        # self.state[0:self.H.size] = 100 * (self.H_reshaped)/(np.max(self.H_reshaped)) # Normalizing the value for the neural network (avoiding errors)
        # self.state[self.H.size:self.H.size + self.loc_users_t.size] = self.loc_users_reshaped #SHOULDN't WE ALSO NORMALIZE THIS TO /1000?
        # self.state = self.state / np.max(self.state) #  the last normalization!
        return self.state

