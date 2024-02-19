import numpy as np
#import networkx as nx

class Specifications:
    def __init__(self, USER_NO, SLICE_NO, CONST_D_MAX, CONST_R_MIN, PACKET_SIZE):
        self.USER_NO = USER_NO
        self.SLICE_NO = SLICE_NO

        self.mat_specs = np.zeros([USER_NO, 4]) #np.zeros([USER_NO, 6, VNF_NO]) # 6 specifications # Why VNF number is in this?
        # Each slice type gets the determined random values obtained by user_selected_slice; we can have repetetive values in list_tolerable_delay, list_min_rate, and list_order_of_isolation

        #self.user_tolerable_delay = 100  # In ms
        if CONST_D_MAX == -1:
            self.list_tolerable_delay = [10, 20, 30, 40, 50] #[100, 100, 100, 100, 100] # [50, 70, 90, 110, 130] # in ms (to be updated after consulting with Faouzi)
        else:
            self.list_tolerable_delay = [CONST_D_MAX] * SLICE_NO #[CONST_D_MAX, CONST_D_MAX, CONST_D_MAX, CONST_D_MAX, CONST_D_MAX] # dependent on SLICE_NO

        #self.user_min_rate = 1  # In Mbps (former case)
        if CONST_R_MIN == -1:
            self.list_min_rate = [1, 5, 10, 20, 30] #[1, 1, 1, 1, 1] # [1, 3, 5, 10, 20] # In Mbps (to be updated after consulting with Faouzi)
        else:
            self.list_min_rate = [CONST_R_MIN] * SLICE_NO

        if PACKET_SIZE == -1:
            self.packet_size_multipled_by_packet_no = [0.01, 0.1, 0.3, 0.7, 1.2] # [0.01, 0.05, 0.1, 0.3, 0.5] # data rate (in Mbps)# it is not per second; it is per timeslot length (~100ms)
        else:
            self.packet_size_multipled_by_packet_no = [PACKET_SIZE] * SLICE_NO
    def _(self):
        for u in range(self.USER_NO):
            user_selected_slice = np.random.randint(0, self.SLICE_NO) # choosing the slice type for the user randomly
            self.mat_specs[u, 0] = user_selected_slice  # Type of slices
            self.mat_specs[u, 1] = self.list_min_rate[user_selected_slice] # self.list_min_rate[np.random.randint(len(self.list_min_rate))] #self.user_min_rate * (1 + np.random.rand())  # reqiured bandwith
            self.mat_specs[u, 2] = self.list_tolerable_delay[user_selected_slice] # self.user_tolerable_delay   #self.list_tolerable_delay[np.random.randint(len(self.list_tolerable_delay))] #  tolrable delay
            self.mat_specs[u, 3] = self.packet_size_multipled_by_packet_no[user_selected_slice]  # Pacekt_size * packet_no (in Mb per timeslot length)
        return self.mat_specs