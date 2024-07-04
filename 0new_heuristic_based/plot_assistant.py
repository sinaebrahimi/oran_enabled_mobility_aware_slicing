# %% Imports
import numpy as np
import matplotlib.pyplot as plt
# -----

#%% Graph plotting:
##########
def plot_graph(title, data, labels, colors, linestyles, x_label, y_label):
    print("-------------\n{}:\n-------------\n".format(title))
    ####plt.ion()  # Turn on interactive mode
    if plt.isinteractive():
        plt.clf()  # Clear the current figure
        for d, l, c, ls in zip(data, labels, colors, linestyles):
            plt.plot(d, color=c, linestyle=ls, label=l)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.pause(0.001)  # Pause for a short period
        plt.ioff()  # Turn off interactive mode
        plt.show()
    else:
        for d, l, c, ls in zip(data, labels, colors, linestyles):
            plt.plot(d, color=c, linestyle=ls, label=l)
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
######