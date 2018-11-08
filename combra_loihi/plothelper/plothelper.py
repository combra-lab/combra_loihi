import matplotlib.figure
from matplotlib import pyplot as plt
import numpy as np

"""
This is the plot helper toolbox for Loihi SNN
"""


def SavePlot(figure: matplotlib.figure.Figure, directory: str, name: str, filetype: str):
    """
    Save matplotlib figure to a file

    :param figure: matplotlib figure
    :param directory: directory to the file
    :param name: name of the file
    :param filetype: file type of saved figure (only support png and svg)
    :return:
    """
    fileName = directory + name
    print("Plot " + name + " is save to file: " + fileName + ".")
    if filetype == 'svg':
        figure.savefig(fileName + '.svg', format='svg')
    elif filetype == 'png':
        figure.savefig(fileName + '.png', format='png')
    else:
        print("File type " + filetype + " is not supported by PlotHelper.")


def MultiRowVoltagePlot(name: str, directory: str, data: np.ndarray, filetype: str):
    """
    Plot multiple rows of voltage data for each compartment separately

    :param name: name of the figure
    :param directory: directory to the file
    :param data: data of the figure from probe
    :param filetype: file type of saved figure
    :return: figure: matplotlib figure
    """
    if type(data) == list:
        data = np.array(data).reshape(1, len(data))
    row_num = data.shape[0]
    col_num = data.shape[1]
    x_list = np.arange(col_num)
    figure_size = (col_num / 500., row_num * 2)
    figure, ax = plt.subplots(row_num, 1, sharex='col', figsize=figure_size)
    if row_num == 1:
        ax = [ax]
    for num in range(row_num):
        ax[row_num - 1 - num].plot(x_list, data[num, :])
        ax[row_num - 1 - num].set_ylabel(str(num))
    ax[0].set_title(name)
    plt.xlabel("Simulation time (ms)")
    SavePlot(figure, directory, name, filetype)
    return figure


def MultiRowCurrentPlot(name: str, directory: str, data: np.ndarray, filetype: str):
    """
    Plot multiple rows of current data for each compartment separately

    :param name: name of the figure
    :param directory: directory to the file
    :param data: data of the figure from probe
    :param filetype: file type of saved figure
    :return: figure: matplotlib figure
    """
    figure = MultiRowVoltagePlot(name, directory, data, filetype)
    return figure


def FiringRateCompute(data: np.ndarray, window: int):
    """
    Compute firing rate of single or multiple neurons using sliding window

    :param data: data of neuron spikes
    :param window: window size in ms
    :return: fr_data: data of firing rates
    :return: fr_x: x axis of firing rates
    """
    if type(data) == list:
        data = np.array(data).reshape(1, len(data))
    row_num = data.shape[0]
    col_num = data.shape[1]
    fr_data = np.zeros((row_num, col_num - window))
    for num in range(col_num-window):
        fr_data[:, num] = data[:, num:num+window].sum(axis=1) / (window / 1000.)
    fr_x = np.arange(col_num - window) + int(window / 2)
    return fr_data, fr_x


def FiringRatePlot(name: str, directory: str, data: np.ndarray, filetype: str, window=250):
    """
    Plot firing rate of spike data

    :param name: name of the figure
    :param directory: directory to the file
    :param data: data of the figure in neuron spikes
    :param filetype: file type of saved figure
    :param window: window size in ms
    :return: figure: matplotlib figure
    """
    if type(data) == list:
        data = np.array(data).reshape(1, len(data))
    row_num = data.shape[0]
    col_num = data.shape[1]
    if col_num < window:
        window = int(col_num / 4)
    fr_data, fr_x = FiringRateCompute(data, window)
    figure_size = (col_num / 500., row_num * 2)
    figure, ax = plt.subplots(row_num, 1, sharex='col', figsize=figure_size)
    if row_num == 1:
        ax = [ax]
    for num in range(row_num):
        ax[row_num - 1 - num].plot(fr_x, fr_data[num, :])
        ax[row_num - 1 - num].set_ylabel(str(num))
    ax[0].set_title(name)
    plt.xlabel("Simulation time (ms)")
    SavePlot(figure, directory, name, filetype)
    return figure


def SpikeTime2Spikes(spike_times: list, time_steps):
    """
    Transform spike times to spikes of each time step in a ndarray

    :param spike_times: time of spikes
    :param time_steps: number of time steps
    :return: spike_date: ndarray of spikes
    """
    row_num = len(spike_times)
    spike_data = np.zeros((row_num, time_steps))
    for num in range(row_num):
        spike_data[num, spike_times[num]] = 1
    spike_data = np.int_(spike_data)
    return spike_data
