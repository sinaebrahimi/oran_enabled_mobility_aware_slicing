import numpy as np
from scipy.stats import rayleigh
from scipy.spatial import distance
import random
import matplotlib.pyplot as plt

class Location:
    def __init__(self, BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY, X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY):
        self.mat_bs_loc = np.zeros([BS_NO, 2])
        self.mat_du_loc = np.zeros([DU_NO, 2])
        self.BS_NO = BS_NO
        self.DU_NO = DU_NO
        self.RU_PER_DU_NO = RU_PER_DU_NO
        self.FH_BW_CAPACITY = FH_BW_CAPACITY
        self.E2_BW_CAPACITY = E2_BW_CAPACITY
        self.PRB_NO = PRB_NO
        self.USER_NO = USER_NO
        self.X_LIM = X_LIM
        self.eta = ETA_AREA  # Power of additive white Gaussian noise (AWGN)
        self.scale = RAYLEIGH_SCALE  # Sigma in Rayleigh distribution formula
        self.V = np.zeros([USER_NO])  # Velocity for each user
        self.angle = np.zeros([USER_NO])  # Angle in degrees
        self.distances_du_ru = None  # Distance between DUs and RUs
        self.distances_ric_du = None  # Distance between RIC and DUs
        mean_angle = 0  # Mean angle
        std_dev_angle = 30  # Standard deviation of angle
        for u in range(self.USER_NO):
            angle_noise = np.random.normal(mean_angle, std_dev_angle)
            angle = angle_noise if abs(angle_noise) <= 90 else np.sign(angle_noise) * (180 - abs(angle_noise))
            self.angle[u] = angle
            if VELOCITY == -1:
                self.V[u] = random.uniform(0, 40)
            else:
                self.V[u] = VELOCITY

    def bs_location(self):
        cells_x = int(np.sqrt(self.BS_NO))
        cells_y = int(np.ceil(self.BS_NO / cells_x))
        cell_width = self.X_LIM / cells_x
        cell_height = self.X_LIM / cells_y
        bs_count = 0
        for i in range(cells_x):
            for j in range(cells_y):
                if bs_count < self.BS_NO:
                    bs_x = (i + 0.5) * cell_width
                    bs_y = (j + 0.5) * cell_height
                    self.mat_bs_loc[bs_count, 0] = bs_x
                    self.mat_bs_loc[bs_count, 1] = bs_y
                    bs_count += 1
                else:
                    break
        return self.mat_bs_loc

    def du_location(self):
        cells_x = int(np.sqrt(self.DU_NO))
        cells_y = int(np.ceil(self.DU_NO / cells_x))
        cell_width = self.X_LIM / cells_x
        cell_height = self.X_LIM / cells_y
        du_count = 0
        for i in range(cells_x):
            for j in range(cells_y):
                if du_count < self.DU_NO:
                    du_x = (i + 0.5) * cell_width
                    du_y = (j + 0.5) * cell_height
                    self.mat_du_loc[du_count, 0] = du_x
                    self.mat_du_loc[du_count, 1] = du_y
                    du_count += 1
                else:
                    break
        return self.mat_du_loc

    def du_ru_distance(self):
        # Calculate distances between DUs and RUs
        self.distances_du_ru = distance.cdist(self.mat_du_loc, self.mat_bs_loc)
        return self.distances_du_ru

    def ric_du_distance(self):
        # Calculate distances between RIC and DUs
        self.distances_ric_du = distance.cdist(np.array([[0, 0]]), self.mat_du_loc)
        return self.distances_ric_du

    def adj_matrix(self):
        # Calculate distances if not already calculated
        if self.distances_du_ru is None:
            self.du_ru_distance()
        if self.distances_ric_du is None:
            self.ric_du_distance()

        # Initialize adjacency matrices with zeros
        du_ru_adj_matrix = np.zeros((self.DU_NO, self.BS_NO))
        ric_du_adj_matrix = np.ones((1, self.DU_NO))  # Connect RIC to all DUs

        # Find the closest RUs for each DU
        closest_ru_indices = np.argsort(self.distances_du_ru, axis=1)[:, :self.RU_PER_DU_NO]

        # Connect DUs to the closest RUs
        for i in range(self.DU_NO):
            du_ru_adj_matrix[i, closest_ru_indices[i]] = 1

        return du_ru_adj_matrix, ric_du_adj_matrix

    def links_capacity(self):
        du_ru_adj_matrix, ric_du_adj_matrix = self.adj_matrix()
        mat_fh_links_capacity = du_ru_adj_matrix * self.FH_BW_CAPACITY
        mat_e2_links_capacity = ric_du_adj_matrix * self.E2_BW_CAPACITY
        return mat_fh_links_capacity, mat_e2_links_capacity

    
    def visualize_ru_du_locations(self, du_ru_adj_matrix):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot DUs with assigned colors
        colors = plt.cm.viridis(np.linspace(0, 1, self.DU_NO))
        for du_index in range(self.DU_NO):
            du_x, du_y = self.mat_du_loc[du_index]
            ax.scatter(du_x, du_y, color=colors[du_index], marker='x', s=100)
            ax.text(du_x, du_y, str(du_index), color='black', ha='center', va='center')

        # Plot RUs with square markers and connected DUs' colors
        for du_index in range(self.DU_NO):
            connected_ru_indices = np.where(du_ru_adj_matrix[du_index] == 1)[0]
            for ru_index in connected_ru_indices:
                ru_x, ru_y = self.mat_bs_loc[ru_index]
                ax.scatter(ru_x, ru_y, color=colors[du_index], marker='s', s=100)
                ax.text(ru_x, ru_y, str(ru_index), color='white', ha='center', va='center')

        # Set labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Location of DUs and RUs')

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
        self.handover_prediction = np.zeros(self.USER_NO, dtype=bool)  # Initialize boolean array for handover prediction
        self.mat_b_connected = np.zeros(self.USER_NO) 
        self.mat_b_pred_connected = np.zeros(self.USER_NO) 

        for u in range(self.USER_NO):
            if t == 0:
                loc_user[t, u, :] = self.X_LIM * np.random.rand(2)
            else:
                next_pos = loc_user[t-1, u, :] + [self.V[u] * np.cos(np.radians(self.angle[u])),
                                                self.V[u] * np.sin(np.radians(self.angle[u]))]
                loc_user[t, u, :] = np.clip(next_pos, 0, self.X_LIM)

            next_pos_pred = loc_user[t, u, :] + [self.V[u] * np.cos(np.radians(self.angle[u])),
                                                self.V[u] * np.sin(np.radians(self.angle[u]))]
            loc_user_pred[t, u, :] = np.clip(next_pos_pred, 0, self.X_LIM)

            for b in range(self.BS_NO):
                x_b, y_b = self.mat_bs_loc[b]
                x_u, y_u = loc_user[t, u]
                x_u_pred, y_u_pred = loc_user_pred[t, u]
                distance_user_bs = distance.euclidean((x_b, y_b), (x_u, y_u)) or 1
                distance_user_bs_pred = distance.euclidean((x_b, y_b), (x_u_pred, y_u_pred)) or 1

                self.mat_distance[b, u] = distance_user_bs
                self.mat_distance_pred[b, u] = distance_user_bs_pred
                # ------------------------------------
                d_alpha = distance_user_bs**(-self.eta) #path loss
                d_pred_alpha = distance_user_bs_pred**(-self.eta) # not used
                # ------------------------------------
                H_u = rayleigh.rvs(scale=self.scale, size=self.PRB_NO) #size=o_d.size) # added the sigma (some controlling parameter in rayleigh distribution to have more variance in the values)
                H_u *= d_alpha
                H_u_pred = rayleigh.rvs(scale=self.scale, size=self.PRB_NO)
                H_u_pred *= d_pred_alpha
                # H_u = rayleigh.rvs(o_d) # calculating user's channel gain using Rayleigh distribution
                self.H[b, :, u] = H_u
                self.H_pred[b, :, u] = H_u_pred # np.mean(H_u) # mean channel gain for estimating the next timeslot #Maybe not too good!

            self.b_connected = self.mat_distance[:, u].argmin()
            self.mat_b_connected[u] = self.b_connected
            self.associator[u, self.b_connected] = 1

            self.b_pred_connected = self.mat_distance_pred[:, u].argmin()
            self.mat_b_pred_connected[u] = self.b_pred_connected
            self.associator_pred[u, self.b_pred_connected] = 1

            if self.b_pred_connected != self.b_connected:
                self.handover_prediction[u] = True

        return loc_user, loc_user_pred, self.H, self.H_pred, self.associator, self.associator_pred, self.mat_distance, self.mat_distance_pred, self.handover_prediction, self.mat_b_connected, self.mat_b_pred_connected
    
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
