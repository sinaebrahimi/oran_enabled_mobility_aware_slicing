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
    def user_association(self):
        self.e0 = 0
        self.e1 = self.USER_NO * self.BS_NO
        # self.e1 = self.USER_NO # * self.BS_NO
        self.temp_chi = (self.action[self.e0:self.e1])
        # self.temp_chi = (self.action[self.e0:self.e1]+1)/2 #transition from [-1,1] to [0,1]
        # self.temp_chi = np.clip(self.temp_chi, 0, 1)
        self.temp_chi = np.clip(self.temp_chi, 0, 1) # to avoid negative values # to make sure that the values are between 0 and 1
        self.temp_chi_reshaped = np.reshape(self.temp_chi, [self.USER_NO, self.BS_NO])

        connected_users = set()  # Keep track of users already connected

        for u in range(self.USER_NO):
            for b in range(self.BS_NO):
                # Check if user u is already connected to a base station
                if u in connected_users:
                    continue
                # If a user is not connected to any BS, connect them to the BS with the highest connection probability
                max_prob_bs = np.argmax(self.temp_chi_reshaped[u, :])
                self.chi[u, max_prob_bs] = 1
                connected_users.add(u)  # Mark user u as connected to a base station
        self.chi_num = np.argmax(self.chi, axis=1)
        return self.chi_num, self.chi

    def user_association_heuristic(self):
        self.e0 = 0
        self.e1 = self.USER_NO # * self.BS_NO
        self.temp_chi = (self.action[self.e0:self.e1])#transition from [-1,1] to [0,1]
        # self.temp_chi = np.clip(self.temp_chi, 0, 1)
        self.temp_chi = np.clip(self.temp_chi, 0.001, 1) # to avoid negative values # to make sure that the values are between 0 and 1
        self.temp_chi_reshaped = np.reshape(self.temp_chi, [self.USER_NO])

        # Reshape to ensure it's a column vector if it's not already
        temp_chi_vector = self.temp_chi_reshaped.reshape(self.USER_NO, 1)

        # Combine normalized action values with channel gains
        # Broadcasting temp_chi_reshaped across all BS
        weighted_preferences = self.H_b.T

        # Choose the BS with the highest weighted preference for each user
        self.chi_num = np.argmax(weighted_preferences, axis=1)

        self.chi[np.arange(self.USER_NO), self.chi_num] = 1


        # # Count users per base station and check for overloading
        # user_counts = np.sum(self.chi, axis=0)
        # max_allowed_users_per_bs = np.ceil(self.USER_NO / self.BS_NO)
        
        # Reallocate users from overloaded BS
        # for b in range(self.BS_NO):
        #     if user_counts[b] > max_allowed_users_per_bs:
        #         # Find users currently assigned to this overloaded BS
        #         overloaded_users = np.where(self.chi[:, b] == 1)[0]
        #         # Sort these users based on their channel gains to b, ascending order
        #         overloaded_users = overloaded_users[np.argsort(self.H_b[b, overloaded_users])]
        #         for u in overloaded_users:
        #             if user_counts[b] <= max_allowed_users_per_bs:
        #                 break
        #             # Attempt to reallocate to next highest preference not overloaded
        #             other_bs_preferences = weighted_preferences[u, :]
        #             # Set current overloaded BS preference to very low to avoid reselection
        #             other_bs_preferences[b] = -np.inf
        #             new_bs = np.argmax(other_bs_preferences) # choosing the second best BS for the user
        #             if user_counts[new_bs] < max_allowed_users_per_bs:
        #                 # Reassign user u from b to new_bs
        #                 self.chi[u, b] = 0
        #                 self.chi[u, new_bs] = 1
        #                 user_counts[b] -= 1
        #                 user_counts[new_bs] += 1
        
        # Update self.chi_num to reflect the final user-BS associations
        self.chi_num = np.argmax(self.chi, axis=1)

        return self.chi_num, self.chi
    # def fh_e2_remaining_capacity(self):    #SKIPPING FOR NOW    
    #     return self.temp_mat_fh_links_capacity, self.temp_mat_e2_links_capacity
    def ran_prb_allocation_old(self): # Equivalent to \rho^{b}_{o,u}(t) in the paper; PRB allocation
        self.e2 = self.e1 + self.USER_NO
        self.temp_rho = (self.action[self.e1:self.e2]+1)/2 
        self.temp_rho_reshaped = np.reshape(self.temp_rho, [self.USER_NO])
        self.rho_num = np.floor(self.temp_rho_reshaped * self.PRB_NO)
        self.rho_num = np.clip(self.rho_num, 0, self.PRB_NO).astype(int)

        unallocated_PRBs = np.zeros([self.USER_NO])  # flag to store the values of PRBs we failed to give to users

        # # #
        # # Scale normalized actions by the rate requirements from mat_specs
        normalized_rate_requirements = self.mat_specs[:, 1] / np.max(self.mat_specs[:, 1])
        rate_scaled_rho = self.temp_rho * normalized_rate_requirements # scale it with the rate requirements
        self.temp_rho_reshaped = np.reshape(rate_scaled_rho, [self.USER_NO])

        # Calculate minimum PRBs required based on some rate to PRB mapping logic
        min_prbs_per_user = np.ceil(self.mat_specs[:, 1] / np.max(self.mat_specs[:, 1]) * self.PRB_NO / self.USER_NO).astype(int)
        min_prbs_per_user = np.clip(min_prbs_per_user, 1, self.PRB_NO)  # Ensure at least 1 PRB, adjust logic as needed


        self.rho_num = np.floor(self.temp_rho_reshaped * self.PRB_NO)
        # Ensure minimum PRBs are allocated if the initial calculation is too low
        self.rho_num = np.maximum(self.rho_num, min_prbs_per_user)
        self.rho_num = np.clip(self.rho_num, 0, self.PRB_NO).astype(int)
        #self.rho_num = np.clip(self.rho_num, 0, self.PRB_NO -1).astype(int)

        

        # First Phase: Allocate minimum PRBs to each user
        for u in range(self.USER_NO):
            allocated_prbs = 0
            for k in range(self.PRB_NO):
                for b in range(self.BS_NO):
                    if self.chi[u, b] == 1 and allocated_prbs < min_prbs_per_user[u]:
                        if np.sum(self.rho[b, k, :]) == 0:  # Check if PRB is free
                            self.rho[b, k, u] = 1
                            allocated_prbs += 1

        # Second Phase: Allocate remaining PRBs to users
        for u in range(self.USER_NO):
            additional_prbs_needed = self.rho_num[u] - np.sum(self.rho[:, :, u])
            for k in range(self.PRB_NO):
                for b in range(self.BS_NO):
                    if self.chi[u, b] == 1 and additional_prbs_needed > 0:
                        if np.sum(self.rho[b, k, :]) == 0:  # Check if PRB is still free
                            self.rho[b, k, u] = 1
                            additional_prbs_needed -= 1

        # Check final allocations
        for u in range(self.USER_NO):
            if np.sum(self.rho[:, :, u]) < self.rho_num[u]:
                unallocated_PRBs[u] = self.rho_num[u] - np.sum(self.rho[:, :, u])

        # for k in range(self.PRB_NO):
        #     for b in range(self.BS_NO):
        #         for u in range(self.USER_NO):
        #             if self.chi[u, b] == 1:
        #                 if np.sum(self.rho[b, :, u]) < self.rho_num[u]:
        #                     if np.sum(self.rho[b, k, :]) != 1:
        #                         self.rho[b, k, u] = 1

        # for u in range(self.USER_NO):
        #     if np.sum(self.rho[:, :, u]) < self.rho_num[u]:
        #         unallocated_PRBs[u] = self.rho_num[u] - np.sum(self.rho[:, :, u])
        
        return self.temp_rho_reshaped, self.rho_num, self.rho, unallocated_PRBs
    

    def ran_prb_allocation(self): # Equivalent to \rho^{b}_{o,u}(t) in the paper; PRB allocation
        # self.e2 = self.e1 + self.USER_NO
        self.e2 = self.e1 + self.BS_NO * self.PRB_NO * self.USER_NO
        # self.temp_rho = (self.action[self.e1:self.e2]+1)/2 
        self.temp_rho = (self.action[self.e1:self.e2])
        self.temp_rho_reshaped = np.reshape(self.temp_rho, [self.BS_NO, self.PRB_NO, self.USER_NO])

        self.done_user_prb_allocation = 0
        cnt_u = 0 

        for u in range(self.USER_NO):
            for b in range(self.BS_NO):
                if self.chi[u,b] == 1:
                    for k in range(self.PRB_NO):
                        if np.sum(self.rho[b, k, :]) == 0: # is the PRB allocated to another user or not?
                            if self.temp_rho_reshaped[b, k, u] >= 0.5: # binarizing the action for rho
                                self.rho[b, k, u] = 1
            if np.sum(self.rho[:, :, u]) > 0:
                cnt_u += 1

        if cnt_u == self.USER_NO:
            self.done_user_prb_allocation = 1
        return self.done_user_prb_allocation, self.rho, cnt_u

    def ran_prb_allocation_heuristic(self):
        self.e2 = self.e1 + self.USER_NO
        self.temp_rho_reshaped = np.zeros([self.USER_NO])
        unallocated_PRBs = np.zeros([self.USER_NO])

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
                # channel_gain = self.H_b[b, u]
                # Weight is proportional to rate requirement and inversely proportional to channel gain
                # weights[i] = rate_requirement / channel_gain
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

        return self.temp_rho_reshaped, self.rho_num, self.rho, unallocated_PRBs 
    
    def ran_power_allocation_water_filling(self, H, SIGMA_NOISE):
        self.e3 = self.e2 + self.USER_NO
        self.temp_p = (self.action[self.e2:self.e3]+1)/2 ### normalizing the values that are previously between [-1,1] to [0,1]
        self.temp_p = np.clip(self.temp_p, 0.1, 1) # to avoid 0 values for power
        self.temp_p_reshaped = np.reshape(self.temp_p, [self.USER_NO])

        inverse_sinr = SIGMA_NOISE / H # b,k,u dimensions
        for b in range(self.BS_NO):
            users_assigned = np.where(self.chi[:, b] == 1)[0]

            if len(users_assigned) > 0:
                # Normalize DRL output for users assigned to BS b
                rate_requirements = self.mat_specs[users_assigned,1]
                normalized_p = rate_requirements / np.sum(rate_requirements)
                # normalized_p = self.temp_p_reshaped[users_assigned] / np.sum(self.temp_p_reshaped[users_assigned])
                
                # Power allocated to each user in BS b
                user_power_allocation = self.MAX_POWER * normalized_p
                
                # Allocate power to each PRB based on inverse SINR and binary variables
                for u in users_assigned:
                    user_power = user_power_allocation[users_assigned.tolist().index(u)]
                    # Sort PRBs based on inverse SINR for user u in BS b
                    prb_indices = np.where(self.rho[b, :, u] == 1)[0]
                    sorted_prbs = prb_indices[np.argsort(inverse_sinr[b, prb_indices, u])]

                    # Water filling allo_latestcation
                    remaining_power = user_power
                    power_per_prb = remaining_power / len(sorted_prbs)  # Start with equal power distribution

                    for k in sorted_prbs:
                        if remaining_power <= 0:
                            break
                        allocated_power = min(power_per_prb, remaining_power)
                        # allocated_power = max(0, remaining_power - inverse_sinr[b, k, u]) 
                        self.p[b, k, u] = allocated_power
                        remaining_power -= allocated_power

                        # Recalculate power_per_prb for remaining PRBs
                        remaining_prbs = len(sorted_prbs) - np.where(sorted_prbs == k)[0][0] - 1
                        if remaining_prbs > 0:
                            power_per_prb = remaining_power / remaining_prbs

            else:
                # no users assigned to the BS, so, no power allocation needed
                pass


        return self.temp_p_reshaped, self.p_num, self.p 

    
    def ran_power_allocation_without_drl(self): #_without_drl
        self.e3 = self.e2 + self.USER_NO
        self.temp_p = (self.action[self.e2:self.e3]+1)/2 ### normalizing the values that are previously between [-1,1] to [0,1]
        self.temp_p = np.clip(self.temp_p, 0.1, 1) # to avoid 0 values for power
        self.temp_p_reshaped = np.reshape(self.temp_p, [self.USER_NO])

        # Normalize action values to represent proportion of power each user should get
        #normalized_actions = self.temp_p_reshaped / np.sum(self.temp_p_reshaped)
        #self.scale = self.MAX_POWER / self.PRB_NO # This ensures that we'll never exceed MAX_POWER in sum_p

        for b in range(self.BS_NO):
            users_connected = np.where(self.chi_num == b)[0]
            if len(users_connected) > 0:
                # Get the initial power allocations for these users
                temp_power_allocations = self.temp_p_reshaped[users_connected]

                normalized_power = temp_power_allocations / np.sum(temp_power_allocations)
                # Get the rate requirements for these users
                rate_requirements = self.mat_specs[users_connected,1]
                fairness_weights = rate_requirements / np.sum(rate_requirements)

                # adjusted_power = normalized_power * fairness_weights
                adjusted_power = fairness_weights

                # Ensure the total power does not exceed MAX_POWER of the BS
                adjusted_power = adjusted_power / np.sum(adjusted_power) * self.MAX_POWER

                self.p_num[users_connected] = adjusted_power

                for k in range(self.PRB_NO):
                    for i, u in enumerate(users_connected):
                        if self.rho[b, k, u] == 1:
                            self.p[b, k, u] = adjusted_power[i] / np.sum(self.rho[b, :, u])

        # Ensure no user gets zero power unless their initial allocation was zero
        # for u in range(self.USER_NO):
        #     if self.temp_p_reshaped[u] > 0 and self.p_num[u] == 0:
        #         # Redistribute a small amount of power to this user
        #         for b in range(self.BS_NO):
        #             if self.chi_num[u] == b:
        #                 prbs_allocated = np.where(self.rho[b, :, u] == 1)[0]
        #                 if len(prbs_allocated) > 0:
        #                     self.p[b, prbs_allocated[0], u] = min(self.temp_p_reshaped[u], self.MAX_POWER / self.USER_NO)
        return self.temp_p_reshaped, self.p_num, self.p    
    
    def ran_power_allocation(self):
        self.e3 = self.e2 + self.BS_NO * self.PRB_NO * self.USER_NO
        self.temp_p = (self.action[self.e2:self.e3])
        # self.temp_p = (self.action[self.e2:self.e3]+1)/2 ### normalizing the values that are previously between [-1,1] to [0,1]
        self.temp_p = np.clip(self.temp_p, 0.1, 1) # to avoid 0 values for power
        self.temp_p_reshaped = np.reshape(self.temp_p, [self.BS_NO, self.PRB_NO, self.USER_NO])

        self.scale = self.MAX_POWER / self.PRB_NO
        self.done_user_power_allocation = 0
        cnt_u = 0

        for u in range(self.USER_NO):
            for b in range(self.BS_NO):
                if self.chi[u, b] == 1:
                    for k in range(self.PRB_NO):
                        if (self.rho[b, k, u]) == 1:
                            if self.remained_power[b] - (self.scale * self.temp_p_reshaped[b, k, u]) > 0:
                                self.p[b, k, u] = self.scale * self.temp_p_reshaped[b, k, u]
                                self.remained_power[b] -= self.p[b, k, u]

            if np.sum(self.p[:, :, u]) > 0:
                cnt_u += 1

        if cnt_u == self.USER_NO:
            self.done_user_power_allocation = 1

        return self.done_user_power_allocation, self.p, cnt_u
    
    def ran_power_allocation_old(self):
        self.e3 = self.e2 + self.USER_NO
        self.temp_p = (self.action[self.e2:self.e3]+1)/2 ### normalizing the values that are previously between [-1,1] to [0,1]
        self.temp_p = np.clip(self.temp_p, 0.1, 1) # to avoid negative values # to make sure that the values are between 0 and 1
        self.temp_p_reshaped = np.reshape(self.temp_p, [self.USER_NO])
        self.scale = self.MAX_POWER / self.PRB_NO # This ensures that we'll never exceed MAX_POWER in sum_p

        # Calculate minimum PRBs required based on some rate to PRB mapping logic
        min_power_per_user = self.mat_specs[:, 1] / np.max(self.mat_specs[:, 1]) * self.PRB_NO / self.USER_NO


        self.p_num = np.maximum(self.p_num, min_power_per_user) # Ensure minimum power is allocated
        self.p_num = np.clip(self.p_num, 0, self.MAX_POWER) # Ensure maximum power is not exceeded

        for u in range(self.USER_NO):
            self.p_num[u] = self.scale * self.temp_p_reshaped[u]

        # According to the allocations in rho, allocate p_num[u] to p[b,k,u] whenever rho[b,k,u] = 1
        for b in range(self.BS_NO):
            for k in range(self.PRB_NO):
                for u in range(self.USER_NO):
                    if self.rho[b, k, u] == 1: #chi is already checked for rho!
                        self.p[b, k, u] = self.p_num[u]

        return self.temp_p_reshaped, self.p_num, self.p


#%%%%
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
    def __init__(self, H, mat_ssl_u_total, mat_chi, mat_rho, mat_p):
        self.H = H # USER_NO * BS_NO
        self.mat_ssl_u_total = mat_ssl_u_total
        self.mat_chi = mat_chi
        self.mat_rho = mat_rho
        self.mat_p = mat_p
        #to add other states
        # self.mat_specs = mat_specs # s3 = 4 * USER_NO # includes 0: selected slice, 1: min rate, 2: tolerable delay, 3: packet size

    def _(self):
        self.s1 = self.H.size # b*u
        self.s2 = self.s1 + self.mat_ssl_u_total.size
        self.s3 = self.s2 + self.mat_chi.size
        self.s4 = self.s3 + self.mat_rho.size
        self.s5 = self.s4 + self.mat_p.size
        self.states_no = self.s5
        #self.H.size + self.mat_ssl_u_rate.size + self.mat_ssl_u_delay.size + self.mat_chi_compressed.size + self.mat_rho_compressed.size + self.mat_p_compressed.size
        self.state = np.zeros([self.states_no])

        # self.loc_users_reshaped = np.reshape(self.loc_users_t, [self.loc_users_t.size])
        self.H_reshaped = np.reshape(self.H, [self.H.size])
        self.mat_ssl_u_total_reshaped = np.reshape(self.mat_ssl_u_total, [self.mat_ssl_u_total.size])
        self.mat_chi_compressed_reshaped = np.reshape(self.mat_chi, [self.mat_chi.size])
        self.mat_rho_compressed_reshaped = np.reshape(self.mat_rho, [self.mat_rho.size])
        self.mat_p_compressed_reshaped = np.reshape(self.mat_p, [self.mat_p.size])

        # H
        self.state[0:self.s1] = (self.H_reshaped)/(np.max(self.H_reshaped)) # Normalizing the value for the neural network (avoiding errors)
        
        # mat_ssl_u_total
        self.state[self.s1:self.s2] = self.mat_ssl_u_total_reshaped        

        # ACTIONS
        self.state[self.s2:self.s3] = self.mat_chi_compressed_reshaped #  no need to normalize
        self.state[self.s3:self.s4] = self.mat_rho_compressed_reshaped #  no need to normalize
        self.state[self.s4:self.s5] = self.mat_p_compressed_reshaped #  no need to normalize


        # self.state[0:self.loc_users_t.size] = self.loc_users_reshaped / self.X_LIM #Normalizing to /1000?
        # self.state[self.loc_users_t.size:self.loc_users_t.size + self.H.size] = 100 * (self.H_reshaped)/(np.max(self.H_reshaped)) # Normalizing the value for the neural network (avoiding errors)

        # cant understand multiplications to 100

        # self.state[0:self.H.size] = 100 * (self.H_reshaped)/(np.max(self.H_reshaped)) # Normalizing the value for the neural network (avoiding errors)
        # self.state[self.H.size:self.H.size + self.loc_users_t.size] = self.loc_users_reshaped #SHOULDN't WE ALSO NORMALIZE THIS TO /1000?
        self.state = self.state / np.max(self.state) #  the last normalization!
        return self.state

