import numpy as np
#import networkx as nx

class Specifications:
    def __init__(self, USER_NO, SLICE_NO, CONST_D_MAX, CONST_R_MIN, PACKET_SIZE):
        self.USER_NO = USER_NO
        self.SLICE_NO = SLICE_NO

        self.mat_specs = np.zeros([USER_NO, 5]) #np.zeros([USER_NO, 6, VNF_NO]) 
        # Each slice type gets the determined random values obtained by user_selected_slice; we can have repetetive values in list_tolerable_delay, list_min_rate, and list_order_of_isolation

        #self.user_tolerable_delay = 100  # In ms
        if CONST_D_MAX == -1:
            self.list_tolerable_delay = [20, 10, 100] # eMBB, URLLC, mMTC # for the control plane delay in DL (in ms)
            #self.list_tolerable_delay = [10, 20, 30, 40, 50] #[100, 100, 100, 100, 100] # [50, 70, 90, 110, 130] # in ms (to be updated after consulting with Faouzi)
        else:
            self.list_tolerable_delay = [CONST_D_MAX] * SLICE_NO #[CONST_D_MAX, CONST_D_MAX, CONST_D_MAX, CONST_D_MAX, CONST_D_MAX] # dependent on SLICE_NO

        #self.user_min_rate = 1  # In Mbps (former case)
        if CONST_R_MIN == -1:
            self.list_min_rate = [20, 1, 0.02] # eMBB #fromerly 100, URLLC, mMTC # user plane rate in Mbps
            #self.list_min_rate = [1, 5, 10, 20, 30] #[1, 1, 1, 1, 1] # [1, 3, 5, 10, 20] # In Mbps (to be updated after consulting with Faouzi)
        else:
            self.list_min_rate = [CONST_R_MIN] * SLICE_NO

        if PACKET_SIZE == -1:
            self.packet_size = [0.012, 0.000256, 0.0008] # in mbits # eMBB: 1500, URLLC: 32, mMTC: 100 bytes
        else:
            self.packet_size = [PACKET_SIZE] * SLICE_NO
        
        # Convert lists to numpy arrays
        list_min_rate_array = np.array(self.list_min_rate)
        packet_size_array = np.array(self.packet_size)
        # Perform element-wise division
        self.packet_no = np.ceil(list_min_rate_array / packet_size_array) #packet_no per timeslot length # static for now #  it can be poisson like 10 packet per second

    def _(self):
        # Calculating the number of users per slice category
        first_slice_count = self.USER_NO // 2
        second_slice_count = (self.USER_NO - first_slice_count) // 2
        third_slice_count = self.USER_NO - first_slice_count - second_slice_count
        
        for u in range(self.USER_NO):
            if u < first_slice_count:
                user_selected_slice = 0
            elif u < first_slice_count + second_slice_count:
                user_selected_slice = 1
            else:
                user_selected_slice = 2
            # user_selected_slice = np.random.randint(0, self.SLICE_NO) # choosing the slice type for the user randomly
            self.mat_specs[u, 0] = user_selected_slice  # Type of slices
            self.mat_specs[u, 1] = self.list_min_rate[user_selected_slice] # self.list_min_rate[np.random.randint(len(self.list_min_rate))] #self.user_min_rate * (1 + np.random.rand())  # reqiured bandwith
            self.mat_specs[u, 2] = self.list_tolerable_delay[user_selected_slice] # self.user_tolerable_delay   #self.list_tolerable_delay[np.random.randint(len(self.list_tolerable_delay))] #  tolrable delay
            self.mat_specs[u, 3] = self.packet_size[user_selected_slice]  # Pacekt_size # in bits
            self.mat_specs[u, 4] = self.packet_no[user_selected_slice] ###self.list_min_rate[user_selected_slice] / self.packet_size[user_selected_slice] #packet_no per timeslot length # static for now #  it can be poisson like 10 packet per second
        return self.mat_specs