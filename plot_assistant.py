# %% Imports
import numpy as np
import matplotlib.pyplot as plt
# -----

def plot_graph(data, labels, x_label, y_label):
    for d, l in zip(data, labels):
        plt.plot(d, label=l)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

# Define moving average function
def moving_average(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_size:] -
              cumsum_vec[:-window_size]) / window_size
    return ma_vec