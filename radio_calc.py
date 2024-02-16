import numpy as np
from scipy.stats import rayleigh
import random


class Location:
    def __init__(self, BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY, X_LIM, RAYLEIGH_SCALE, ETA_AREA):
        self.mat_bs_loc = np.zeros([BS_NO, 2])
        self.BS_NO = BS_NO
        self.DU_NO = DU_NO
        self.RU_PER_DU_NO = RU_PER_DU_NO
        self.PRB_NO = PRB_NO
        self.USER_NO = USER_NO
        self.X_LIM = X_LIM
        self.eta = ETA_AREA # power of additive white Gaussian noise (AWGN) #eta in paper # 3 for rural areas
        self.scale = RAYLEIGH_SCALE #2 # sigma in rayleigh distribution formula. 1 by default
        self.V = np.zeros([USER_NO]) #each user have a velocity 
        for u in range(self.USER_NO):
            if VELOCITY == -1:
                self.V[u] = random.uniform(0, 40)
            else:
                self.V[u] = 35 # 35 m/s as default/constant speed for everyone
        self.angle = np.zeros([USER_NO])  # angle in degrees
        self.angle[u] = random.uniform(0, 360)  # Random angle between 0 and 360 degrees


    def bs_location(self): #BS Locations:[[125. 625.] [250. 750.] [375. 875.] [625. 625.] [750. 750.] [875. 875.]]
        # Place RUs evenly within each region
        region_size = self.X_LIM / self.DU_NO # 1000/2 = 500
        for du_idx in range(self.DU_NO):
            # Determine the starting x-coordinate of the region
            region_start_x = du_idx * region_size            
            # Determine the ending x-coordinate of the region
            #region_end_x = (du_idx + 1) * region_size            
            # Determine the y-coordinate of the region (assuming it covers the entire Y-axis)
            region_start_y = self.X_LIM / self.DU_NO  # Assuming the center of the area is the midpoint of the Y-axis
            # Determine the spacing between RUs within the region
            ru_spacing = region_size / (self.RU_PER_DU_NO + 1)
            # Place RUs evenly within the region
            for ru_idx in range(self.RU_PER_DU_NO):
                # Determine the x-coordinate of the RU
                ru_x = region_start_x + ru_spacing * (ru_idx + 1)
                ru_y = region_start_y + ru_spacing * (ru_idx + 1)
                # Store the RU location in the matrix
                self.mat_bs_loc[du_idx * self.RU_PER_DU_NO + ru_idx] = [ru_x, ru_y]
        return self.mat_bs_loc

    def user_location(self, t, loc_user): #also calculates channel gain
        self.mat_bs_loc = Location.bs_location(self)
        self.H = np.zeros([self.BS_NO, self.PRB_NO, self.USER_NO])
        self.H_pred = np.zeros([self.BS_NO, self.PRB_NO, self.USER_NO]) # mean channel gain for estimating the next timeslot
        self.associator = np.zeros([self.USER_NO, self.BS_NO])
        self.associator_pred = np.zeros([self.USER_NO, self.BS_NO])
        self.mat_distance = np.zeros([self.BS_NO, self.USER_NO])
        self.mat_distance_pred = np.zeros([self.BS_NO, self.USER_NO])
        loc_user_pred = np.copy(loc_user)
        # for u in range(self.USER_NO):
        #     if t == 0:
        #         loc_user[t, u, :] = self.X_LIM * np.random.rand()  # initiate user location with a random location in the straight road
        #         loc_user_pred[t, u, :] = loc_user[t, u, :] + self.V
        #         # self.mat_user_loc[t,u,1]=X_LIM*np.random.rand()  + index_height # index_height = 100
        #     else:
        #         if np.sum(loc_user[t-1, u, :])/2 + self.V < self.X_LIM: #checking if we are not out of bounds
        #             loc_user[t, u, :] = loc_user[t-1, u, :] + self.V
        #             if np.sum(loc_user[t, u, :])/2 + self.V < self.X_LIM:
        #                 loc_user_pred[t, u, :] = loc_user[t, u, :] + self.V
        #             else:
        #                 loc_user_pred[t, u, :] = 1
        #         else:
        #             loc_user[t, u, :] = 1
        #             loc_user_pred[t, u, :] = loc_user[t, u, :] + self.V

        for u in range(self.USER_NO):
            if t == 0:
                loc_user[t, u, :] = self.X_LIM * np.random.rand(2)  # initiate user location with a random location in the area
                loc_user_pred[t, u, :] = loc_user[t, u, :] + [self.V[u] * np.cos(np.radians(self.angle[u])), self.V[u] * np.sin(np.radians(self.angle[u]))]
            else:
                # Calculate next position based on velocity and angle, ensuring it stays within bounds
                next_pos = loc_user[t-1, u, :] + [self.V[u] * np.cos(np.radians(self.angle[u])), self.V[u] * np.sin(np.radians(self.angle[u]))]
                loc_user[t, u, :] = np.clip(next_pos, 0, self.X_LIM)
                
                next_pos_pred = loc_user_pred[t, u, :] + [self.V[u] * np.cos(np.radians(self.angle[u])), self.V[u] * np.sin(np.radians(self.angle[u]))]
                loc_user_pred[t, u, :] = np.clip(next_pos_pred, 0, self.X_LIM)

            self.mem_b = []
            self.mem_b_pred = []
            for b in range(self.BS_NO):
                x_b = self.mat_bs_loc[b, 0]
                y_b = self.mat_bs_loc[b, 1]
                # ----------------------------
                x_u = loc_user[t, u, 0]
                y_u = loc_user[t, u, 1]
                x_u_pred = loc_user_pred[t, u, 0]
                y_u_pred = loc_user_pred[t, u, 1]
                # ----------------------------
                d2 = (x_b - x_u)**2 + (y_b - y_u)**2
                d2_pred = (x_b - x_u_pred)**2 + (y_b - y_u_pred)**2
                if d2 == 0:
                    d2 = 1
                if d2_pred == 0:
                    d2_pred = 1

                d = d2**.5 # user distance from the BS
                d_pred = d2_pred**.5 # user distance from the BS (predicted value for next timeslot)
                self.mat_distance[b, u] = d #d2**.5  #d
                self.mat_distance_pred[b, u] = d_pred #d2_pred**.5 # d_pred
                # ------------------------------------
                d_alpha = d**(-self.eta) #path loss
                d_pred_alpha = d_pred**(-self.eta) # not used
                # ------------------------------------
                o_d = np.ones(self.PRB_NO) * d_alpha # np.ones(PRB_NO)*d_alpha
                o_d_pred = np.ones(self.PRB_NO) * d_pred_alpha
                H_u = rayleigh.rvs(scale=self.scale, size=o_d.size) # added the sigma (some controlling parameter in rayleigh distribution to have more variance in the values)
                H_u_pred = rayleigh.rvs(scale=self.scale, size=o_d_pred.size)
                # H_u = rayleigh.rvs(o_d) # calculating user's channel gain using Rayleigh distribution
                self.H[b, :, u] = H_u
                self.H_pred[b, :, u] = H_u_pred # np.mean(H_u) # mean channel gain for estimating the next timeslot #Maybe not too good!
                self.mem_b.append(d)
                self.mem_b_pred.append(d_pred)

            self.mem_b = np.array(self.mem_b)
            self.b_connected = self.mem_b.argmin()
            self.associator[u, self.b_connected] = 1

            self.mem_b_pred = np.array(self.mem_b_pred)
            self.b_pred_connected = self.mem_b_pred.argmin()
            self.associator_pred[u, self.b_pred_connected] = 1

        return loc_user, loc_user_pred, self.H, self.H_pred, self.associator, self.associator_pred, self.mat_distance, self.mat_distance_pred
# %%

class RateCalculation:
    def __init__(self, P, rho, H, associator, BS_NO, PRB_NO, USER_NO, SIGMA_NOISE, BW):
        self.BS_NO = BS_NO
        self.PRB_NO = PRB_NO
        self.USER_NO = USER_NO
        self.SIGMA_NOISE = SIGMA_NOISE
        self.BW = BW
        self.P = P
        self.rho = rho
        self.H = H
        self.mat_rate = np.zeros([self.USER_NO])
        self.mat_rate_prb = np.zeros([self.PRB_NO, self.USER_NO])
        self.associator = associator

    def calculate_interference(self, b, k, u): #BS, PRB, user
        I_inter = 0 # inter-cell interference
        for bb in range(self.BS_NO):
            if bb != b:
                for uu in range(self.USER_NO):
                    if uu != u:
                        I_inter += self.P[bb, k, uu] * self.rho[bb, k, uu] * self.H[bb, k, u]
        return I_inter

    def _(self):
        for u in range(self.USER_NO):
            for b in range(self.BS_NO):
                if self.associator[u, b] == 1:
                    for k in range(self.PRB_NO):
                        I_inter = RateCalculation.calculate_interference(self, b, k, u)
                        self.ph = self.P[b, k, u] * self.rho[b, k, u] * self.H[b, k, u]
                        self.SINR = (self.ph) / (I_inter + self.SIGMA_NOISE) #10e-10 for SIGMA_NOISE
                        self.rate_prb = self.BW * np.log2(1 + self.SINR)
                        self.mat_rate_prb[k, u] = self.rate_prb
                        self.mat_rate[u] += self.rate_prb

        self.mat_rate = self.mat_rate / 10e6 # Convert to Mbps
        return self.mat_rate, self.mat_rate_prb
