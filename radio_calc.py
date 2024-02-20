import numpy as np
from scipy.stats import rayleigh
from scipy.spatial import distance
import random
import matplotlib.pyplot as plt



class Location:
    def __init__(self, BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY, X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY):
        self.X_LIM = X_LIM
        self.CU_NO = 1
        self.mat_cu_ric_loc = np.zeros([self.CU_NO, 2])
        self.mat_cu_ric_loc[0, :] = X_LIM / 2 # (500,500) in our assumptions it is in the middle
        self.mat_bs_loc = np.zeros([BS_NO, 2])
        self.mat_du_loc = np.zeros([DU_NO, 2])
        self.distances_ric_du = np.zeros([DU_NO])
        self.distances_du_ru = np.zeros([DU_NO, BS_NO])
        self.BS_NO = BS_NO
        self.DU_NO = DU_NO
        self.RU_PER_DU_NO = RU_PER_DU_NO        
        self.du_ru_adj_matrix = np.zeros((self.DU_NO, self.BS_NO))# Create an adjacency matrix to connect RUs to DUs based on distances
        self.ric_du_adj_matrix = np.zeros((1, self.DU_NO))
        self.mat_fh_links_capacity = np.zeros((self.DU_NO, self.BS_NO))
        self.mat_e2_links_capacity = np.zeros((1, self.DU_NO))    
        self.FH_BW_CAPACITY = FH_BW_CAPACITY
        self.E2_BW_CAPACITY = E2_BW_CAPACITY
        self.PRB_NO = PRB_NO
        self.USER_NO = USER_NO        
        self.eta = ETA_AREA # power of additive white Gaussian noise (AWGN) #eta in paper # 3 for rural areas
        self.scale = RAYLEIGH_SCALE #2 # sigma in rayleigh distribution formula. 1 by default
        self.V = np.zeros([USER_NO]) #each user have a velocity
        self.angle = np.zeros([USER_NO])  # angle in degrees
        # Define parameters for normal distribution
        mean_angle = 0  # Mean angle
        std_dev_angle = 30  # Standard deviation of angle
        for u in range(self.USER_NO):
            # Generate random angle with normal distribution
            angle_noise = np.random.normal(mean_angle, std_dev_angle)
            # Adjust angle to stay close to 0
            angle = angle_noise if abs(angle_noise) <= 90 else np.sign(angle_noise) * (180 - abs(angle_noise))
            self.angle[u] = angle
            #BEFORE: self.angle[u] = random.uniform(0, 360)  # Random angle between 0 and 360 degrees
            if VELOCITY == -1:
                self.V[u] = random.uniform(0, 40)
            else:
                self.V[u] = VELOCITY # 35 m/s as default/constant speed for everyone


    def bs_location(self):
        # Determine the number of cells/regions in each dimension
        cells_x = int(np.sqrt(self.BS_NO))  # Assuming nearly square grid
        cells_y = int(np.ceil(self.BS_NO / cells_x))
        
        # Calculate the size of each cell
        cell_width = self.X_LIM / cells_x
        cell_height = self.X_LIM / cells_y

        # Place BSs evenly within the grid
        bs_count = 0
        for i in range(cells_x):
            for j in range(cells_y):
                if bs_count < self.BS_NO:
                    # Calculate the coordinates of the BS within the cell
                    bs_x = (i + 0.5) * cell_width  # Place at the center of the cell
                    bs_y = (j + 0.5) * cell_height
                    self.mat_bs_loc[bs_count, 0] = bs_x
                    self.mat_bs_loc[bs_count, 1] = bs_y
                    bs_count += 1
                else:
                    break
        return self.mat_bs_loc
    
    def du_location(self):
        #assuming the number of DUs is lower than the number of RUs (BSs)
        # Determine the number of cells/regions in each dimension
        cells_x = int(np.sqrt(self.DU_NO))  # Assuming nearly square grid
        cells_y = int(np.ceil(self.DU_NO / cells_x))
        
        # Calculate the size of each cell
        cell_width = self.X_LIM / cells_x
        cell_height = self.X_LIM / cells_y

        # Place DUs evenly within the grid
        du_count = 0
        for i in range(cells_x):
            for j in range(cells_y):
                if du_count < self.DU_NO:
                    # Calculate the coordinates of the DU within the cell
                    du_x = (i + 0.5) * cell_width  # Place at the center of the cell
                    du_y = (j + 0.5) * cell_height
                    self.mat_du_loc[du_count, 0] = du_x
                    self.mat_du_loc[du_count, 1] = du_y
                    du_count += 1
                else:
                    break
        return self.mat_du_loc    
    
    def ric_du_distance(self):
        ric_x, ric_y = self.mat_cu_ric_loc[0]
        self.distances_ric_du = [distance.euclidean((ric_x, ric_y), (du_x, du_y)) for du_x, du_y in self.mat_du_loc]
        return self.distances_ric_du
    
    def du_ru_distance(self):
        for i in range(self.DU_NO):
            du_x, du_y = self.mat_du_loc[i]
            # Calculate distances between DU and RUs
            distances = [distance.euclidean((du_x, du_y), (bs_x, bs_y)) for bs_x, bs_y in self.mat_bs_loc]
            self.distances_du_ru[i, :] = distances
        return self.distances_du_ru

    def adj_matrix(self):        
        for i in range(self.DU_NO):
            du_x, du_y = self.mat_du_loc[i]
            # Calculate distances between DU and RUs
            distances = [distance.euclidean((du_x, du_y), (bs_x, bs_y)) for bs_x, bs_y in self.mat_bs_loc]
            # Find the index of the closest RU to the DU
            closest_ru_index = np.argmin(distances)
            self.du_ru_adj_matrix[i, closest_ru_index] = 1
        # Ensure all RUs are assigned to a DU
        while not np.all(self.du_ru_adj_matrix.sum(axis=0) == 1):
            # Get indices of unassigned RUs
            unassigned_indices = np.where(self.du_ru_adj_matrix.sum(axis=0) == 0)[0]
            for idx in unassigned_indices:
                # Find the index of the closest DU to the unassigned RU
                distances = [distance.euclidean((du_x, du_y), self.mat_bs_loc[idx]) for du_x, du_y in self.mat_du_loc]
                closest_du_index = np.argmin(distances)
                self.du_ru_adj_matrix[closest_du_index, idx] = 1
        
        # Now let's connect RIC (or the CU (assuming there's only one)) to DUs
        for du_idx in range(self.DU_NO):
            self.ric_du_adj_matrix[0, du_idx] = 1 #  all DUs are connected to RIC via E2 interface      
        return self.du_ru_adj_matrix, self.ric_du_adj_matrix

    def links_capacity(self): 
        # link capacities are identical; FH_BW_CAPACITY Mbps for FH links and E2_BW_CAPACITY Mbps for E2 links
        self.adj_matrix_ru_du, self.adj_matrix_ric_du = Location.adj_matrix(self)
        self.mat_fh_links_capacity = self.adj_matrix_ru_du * self.FH_BW_CAPACITY # capacity matrix for FH links between DUs and RUs
        self.mat_e2_links_capacity = self.adj_matrix_ric_du * self.E2_BW_CAPACITY
        return self.mat_fh_links_capacity, self.mat_e2_links_capacity
    
    def plot_ru_du_locations(self): # just visualizing the RUs and DUs # not important for the solution
        # Plotting RUs and DUs
        plt.figure(figsize=(8, 8))
        for i in range(self.DU_NO):
            du_x, du_y = self.mat_du_loc[i]
            bs_indices = np.where(self.du_ru_adj_matrix[i] == 1)[0]
            colors = plt.cm.viridis(np.linspace(0, 1, len(bs_indices)))
            for idx, bs_idx in enumerate(bs_indices):
                bs_x, bs_y = self.mat_bs_loc[bs_idx]
                plt.plot([du_x, bs_x], [du_y, bs_y], color=colors[idx], linestyle='--')
            plt.scatter(du_x, du_y, color='red', label=f'DU {i+1}')
        plt.scatter(self.mat_bs_loc[:, 0], self.mat_bs_loc[:, 1], color='blue', label='RUs')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Locations of RUs and DUs')
        plt.legend()
        plt.grid(True)
        plt.show()        


# User location and channel gain calculations
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
    
    def plot_user_movement(self, loc_user, associator, t):
        # Set up figure and axis for plotting
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot BSs
        bs_colors = ['grey' for _ in range(self.BS_NO)]  # Initialize colors for BSs
        for b in range(self.BS_NO):
            bs_x, bs_y = self.mat_bs_loc[b]
            ax.scatter(bs_x, bs_y, color=bs_colors[b], marker='s', s=100)  # Use a square marker for BSs
            ax.text(bs_x, bs_y, str(b), color='white', ha='center', va='center')  # Print BS index inside its symbol

        # Plot users and connections to BSs
        for u in range(self.USER_NO):
            user_color = plt.cm.viridis(u / self.USER_NO)  # Use a different color for each user
            valid_indices = np.where(loc_user[:, u, 0] != 0)[0]  # Find valid indices where user position is not (0, 0)
            if len(valid_indices) > 1:  # If there are valid indices (excluding the initialization)
                ax.plot(loc_user[valid_indices, u, 0], loc_user[valid_indices, u, 1], color=user_color)  # Plot user trajectory
                bs_index = np.where(associator[u] == 1)[0][0]  # Get the index of the BS assigned to the user
                bs_x, bs_y = self.mat_bs_loc[bs_index]
                if bs_colors[bs_index] == 'grey':
                    bs_colors[bs_index] = plt.cm.viridis(u / self.USER_NO)  # Change color to user's color if BS is occupied
                ax.arrow(loc_user[valid_indices[-2], u, 0], loc_user[valid_indices[-2], u, 1],  # Start of arrow at second last valid timestep
                        loc_user[valid_indices[-1], u, 0] - loc_user[valid_indices[-2], u, 0], loc_user[valid_indices[-1], u, 1] - loc_user[valid_indices[-2], u, 1],  # Arrow direction
                        head_width=10, head_length=15, fc=user_color, ec=user_color, linestyle='dotted')  # Arrow properties

                # Plot a dotted line from user's last valid position to the assigned BS
                ax.plot([loc_user[valid_indices[-1], u, 0], bs_x], [loc_user[valid_indices[-1], u, 1], bs_y], color=user_color, linestyle='dotted')

        # Set labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('User Movement Over Time (t='+str(t)+')')

        # Show legend
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(u / self.USER_NO), markersize=10, label=f'User {u}') for u in range(self.USER_NO)]
        legend_handles.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='grey', markersize=10, label='Unoccupied BS'))
        ax.legend(handles=legend_handles)

        # Show plot
        plt.grid(True)
        plt.show()    
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
        self.SINR_dB = np.zeros([self.USER_NO])
        self.signal_strength_dB = np.zeros([self.USER_NO])
        self.interference_dB = np.zeros([self.USER_NO])
        self.noise_plus_interference_dB = np.zeros([self.USER_NO])
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
                        self.interference_dB[u] = 10 * np.log10(I_inter)
                        self.signal_strength = self.P[b, k, u] * self.rho[b, k, u] * self.H[b, k, u]
                        self.signal_strength_dB[u] = 10 * np.log10(self.signal_strength)
                        self.noise_plus_interference = (I_inter + (self.SIGMA_NOISE * self.BW))
                        self.noise_plus_interference_dB = 10 * np.log10(self.noise_plus_interference)
                        self.SINR = self.signal_strength / self.noise_plus_interference #10e(-12.4) for SIGMA_NOISE
                        self.SINR_dB[u] = 10 * np.log10(self.SINR)
                        self.rate_prb = self.BW * np.log2(1 + self.SINR)
                        self.mat_rate_prb[k, u] = self.rate_prb
                        self.mat_rate[u] += self.rate_prb

        self.mat_rate = self.mat_rate / 1e6 # Convert to Mbps
        return self.mat_rate, self.mat_rate_prb, self.SINR_dB, self.signal_strength_dB, self.interference_dB, self.noise_plus_interference_dB
