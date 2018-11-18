"""
MIT License

Copyright (c) 2018 Guangzhi Tang
Copyright (c) 2018 Arpit Shah
Copyright (c) 2018 Computational Brain Lab, Computer Science Department, Rutgers University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import matplotlib.figure
from matplotlib import pyplot as plt
import numpy as np
from nxsdk.utils.plotutils import plotRaster

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


def FiringRateComputeGap(data: np.ndarray):
    """
    Compute firing rate of single or multiple neurons using spike gap time

    :param data: data of neuron spikes
    :return: fr_data: data of firing rates
    :return: fr_x: x axis of firing rates
    """
    if type(data) == list:
        data = np.array(data).reshape(1, len(data))
    row_num = data.shape[0]
    col_num = data.shape[1]
    fr_data = np.zeros((row_num, col_num))
    fr_x = np.arange(col_num)
    spike_times = Spikes2SpikeTime(data)
    for r in range(row_num):
        for t in range(len(spike_times[r]) - 1):
            firing_rate = 1000. / (spike_times[r][t+1] - spike_times[r][t])
            fr_data[r, spike_times[r][t]:spike_times[r][t+1]] = firing_rate
    return fr_data, fr_x


def FiringRatePlot(name: str, directory: str, data: np.ndarray, filetype: str, enable_gap=False, window=250):
    """
    Plot firing rate of spike data

    :param name: name of the figure
    :param directory: directory to the file
    :param data: data of the figure in neuron spikes
    :param filetype: file type of saved figure
    :param enable_gap: if or not using spike gap to compute firing rate
    :param window: window size in ms
    :return: figure: matplotlib figure
    """
    if type(data) == list:
        data = np.array(data).reshape(1, len(data))
    row_num = data.shape[0]
    col_num = data.shape[1]
    if col_num < window:
        window = int(col_num / 4)
    if enable_gap:
        fr_data, fr_x = FiringRateComputeGap(data)
    else:
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


def Spikes2SpikeTime(data: np.ndarray):
    """
    Transform spikes to spike times

    :param data: data of spikes
    :return: spike_times: time of spikes
    """
    if type(data) == list:
        data = np.array(data).reshape(1, len(data))
    spike_times = [np.where(data[num, :])[0].tolist() for num in range(data.shape[0])]
    return spike_times


def SpikesRasterPlot(name: str, directory: str, data: list, sim_time: int, filetype: str):
    """
    Plot Spike Raster

    :param name: name of the figure
    :param directory: directory to the figure
    :param data: spike times of neurons
    :param sim_time: simulation time
    :param filetype: type of file
    :return: figure: matplotlib figure
    """
    figure = plt.figure()
    plt.xlim([0, sim_time])
    plotRaster(data)
    SavePlot(figure, directory, name, filetype)
    return figure
