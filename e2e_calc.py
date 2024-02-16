import numpy as np
import networkx as nx
from initialization import Capacity


class Mapping:
    def __init__(self, action, mat_links_capacity, mat_nodes_and_vms_capacity, mat_specs, associator, USER_NO, VM_NO, VNF_NO, NODE_NO, BS_NO, PRB_NO, MAX_POWER):
        self.USER_NO = USER_NO
        self.VM_NO = VM_NO
        self.VNF_NO = VNF_NO
        self.NODE_NO = NODE_NO
        self.BS_NO = BS_NO
        self.PRB_NO = PRB_NO
        self.MAX_POWER = MAX_POWER
        #---------
        self.mat_placement = np.zeros([self.USER_NO, self.VNF_NO, self.VM_NO, self.NODE_NO])
        self.temp_mat_placement = np.copy(self.mat_placement)
        self.action = action
        self.W_link = np.zeros([self.NODE_NO, self.NODE_NO]) # G graph based on weights(for OSPF-like routing)
        # --------------------------------------------------------------------------
        self.P = np.zeros([self.BS_NO, self.PRB_NO, self.USER_NO])
        self.rho = np.zeros([self.BS_NO, self.PRB_NO, self.USER_NO])
        self.path = np.zeros([self.USER_NO])
        #############################################################
        self.remained_power = np.ones([self.BS_NO]) * self.MAX_POWER
        #############################################################
        self.mat_links_capacity = mat_links_capacity
        self.mat_nodes_and_vms_capacity = mat_nodes_and_vms_capacity
        self.mat_specs = mat_specs
        #############################################################
        self.temp_mat_links_capacity = np.copy(mat_links_capacity)
        self.temp_mat_nodes_and_vms_capacity = np.copy(mat_nodes_and_vms_capacity)
        #############################################################
        self.associator = associator
#%% RAN mapping:
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
                        if np.sum(self.rho[b, k, :]) == 0:
                            if self.temp_rho_reshaped[b, k, u] > .5: # rounding up
                                self.rho[b, k, u] = 1

            if np.sum(self.rho[:, :, u]) > 0:
                cnt_u += 1

        if cnt_u == self.USER_NO:
            self.done_user_prb_allocation = 1

        return self.done_user_prb_allocation, self.rho

    def ran_power_allocation(self):
        self.e2 = self.e1 + self.BS_NO * self.PRB_NO * self.USER_NO
        self.temp_p = (self.action[self.e1:self.e2]+1)/2
        self.temp_p_reshaped = np.reshape(self.temp_p, [self.BS_NO, self.PRB_NO, self.USER_NO])
        self.scale = .01 # What is it?
        self.done_user_power_allocation = 0
        cnt_u = 0
        for u in range(self.USER_NO):
            for b in range(self.BS_NO):
                if self.associator[u, b] == 1:
                    for k in range(self.PRB_NO):
                        if (self.rho[b, k, u]) == 1:
                            if self.remained_power[b] - (self.scale * self.temp_p_reshaped[b, k, u]) > 0:
                                self.P[b, k, u] = self.scale * self.temp_p_reshaped[b, k, u]

            if np.sum(self.P[:, :, u]) > 0:
                cnt_u += 1
        if cnt_u == self.USER_NO:
            self.done_user_power_allocation = 1

        return self.done_user_power_allocation, self.P



#%%%%
class Delay:
    def __init__(self, mat_rate, mat_placement, W_link, mat_links_capacity, mat_nodes_and_vms_capacity, mat_specs, path, associator, mat_distance, USER_NO, VNF_NO, BS_NO):
        self.USER_NO = USER_NO
        self.VNF_NO = VNF_NO
        self.BS_NO = BS_NO
        #-------------------
        self.mat_rate = mat_rate
        self.mat_placement = mat_placement
        self.W_link = W_link
        self.associator = associator
        self.path = path
        self.mat_links_capacity = mat_links_capacity
        self.mat_specs = mat_specs
        self.speed_of_light = 300000000 # 3*10^8 m/s (exactly equal to 299,792,458 metres per second)
        self.mat_distance = mat_distance

        self.mat_delay_proc = np.zeros([self.USER_NO])
        self.mat_delay_tx_ran = np.zeros([self.USER_NO])
        self.mat_delay_prop_ran = np.zeros([self.USER_NO])
        self.mat_delay_tx_cn = np.zeros([self.USER_NO])

    def tx_ran(self):
        for u in range(self.USER_NO):
            # delay_tx_ran = PACKET_SIZE / self.mat_rate[u]
            delay_tx_ran = self.mat_specs[u, 3] / self.mat_rate[u]
            self.mat_delay_tx_ran[u] = delay_tx_ran
        return self.mat_delay_tx_ran

    def prop_ran(self):
        for u in range(self.USER_NO):
            for b in range(self.BS_NO):
                if self.associator[u, b] == 1:
                   d = self.mat_distance[b, u]
                   self.mat_delay_prop_ran[u] = d / self.speed_of_light

        return self.mat_delay_prop_ran


    def _(self):
        self.mat_delay_tot = np.ones([self.USER_NO]) # avoiding zeros to escape division to zero error # to multiply to something smaller as we will multiply it by 1000 to convert to ms
        self.mat_delay_tot *= 0.00001 # 10 us (0.01 ms)
        done_delay_all = 0
        cnt_u = 0
        self.mat_delay_tx_ran = Delay.tx_ran(self)
        self.mat_delay_prop_ran = Delay.prop_ran(self)

        for u in range(self.USER_NO):
            user_tolerable_delay = self.mat_specs[u, 2]
            # if self.mat_delay_tx_cn[u] + self.mat_delay_proc[u] + self.mat_delay_proc[u] < user_tolerable_delay:
            if self.mat_delay_tx_ran[u] + self.mat_delay_prop_ran[u] < user_tolerable_delay:
                cnt_u += 1
                # self.mat_delay_tot[u] = self.mat_delay_tx_cn[u] + self.mat_delay_proc[u] + self.mat_delay_proc[u] why two proc delays?
                self.mat_delay_tot[u] = self.mat_delay_tx_ran[u] + self.mat_delay_prop_ran[u]#mat_delay_tx_ran is significantly greater than others
                #print("User {} total delay: {}".format(u, self.mat_delay_tot[u]))###to remove

        if cnt_u == self.USER_NO:
            done_delay_all = 1

        self.mat_delay_tot = self.mat_delay_tot * 1000   #  To convert to ms
        return done_delay_all, self.mat_delay_tot

# %%
class StateCalculation:
    def __init__(self, H, loc_users_t):
        self.H = H
        self.loc_users_t = loc_users_t
        #to add other states

    def _(self):
        self.states_no = self.H.size + self.loc_users_t.size
        self.state = np.zeros([self.states_no])

        self.H_reshaped = np.reshape(self.H, [self.H.size])
        self.loc_users_reshaped = np.reshape(self.loc_users_t, [self.loc_users_t.size])

        self.state[0:self.H.size] = 100 * (self.H_reshaped)/(np.max(self.H_reshaped)) # Normalizing the value for the neural network (avoiding errors)
        self.state[self.H.size:self.H.size + self.loc_users_t.size] = self.loc_users_reshaped
        self.state = 100 * self.state / np.max(self.state)
        return self.state

