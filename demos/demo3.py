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

import os
import matplotlib as mpl
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import nxsdk.api.n2a as nx
from nxsdk.utils.plotutils import plotRaster
import combra_loihi.api as combra

import numpy as np


def gen_rand_spikes(num_neurons: int, sim_time: int, firing_rate: float):
    """ Generate a random array of shape `(num_neurons, sim_time)` to specify spikes for input.

    :param num_neurons: The number of input neurons
    :param sim_time: Number of millisecond timesteps
    :param firing_rate: General firing rate in Hz (i.e. 10 --> 10 Hz)

    :return: 2D array of binary spike values
    """
    random_spikes = np.random.rand(num_neurons, sim_time) < (firing_rate / 1000.)
    random_spikes = [
        np.where(random_spikes[num, :])[0].tolist()
        for num in range(num_neurons)
    ]
    return random_spikes


if __name__ == '__main__':
    # to see consistent results from run-to-run
    np.random.seed(0)

    net = nx.NxNet()
    sim_time = 8500

    pre_neuron_cnt = 20
    post_neuron_cnt = 20

    # Create pre-synaptic neurons (spike generator)
    pre_synaptic_neurons = net.createSpikeGenProcess(pre_neuron_cnt)
    input_spike_times = gen_rand_spikes(pre_neuron_cnt, sim_time, 10)
    pre_synaptic_neurons.addSpikes(
        spikeInputPortNodeIds=[num for num in range(pre_neuron_cnt)],
        spikeTimes=input_spike_times)

    # Create post-synaptic neuron
    post_neuron_proto = nx.CompartmentPrototype(
        vThMant=10,
        compartmentCurrentDecay=int(1 / 10 * 2 ** 12),
        compartmentVoltageDecay=int(1 / 4 * 2 ** 12),
        functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)
    post_neurons = net.createCompartmentGroup(
        size=post_neuron_cnt, prototype=post_neuron_proto)

    # Create a connection from the pre to post-synaptic neuron
    conn_proto = nx.ConnectionPrototype()
    conn_mask = np.zeros((20, 20))
    # Turn on connections in the mask between the first 10 pre-synaptic and first 10 post-synaptic neurons
    conn_mask[0:10] = np.arange(20) < 10
    # Turn on connections in the mask between the last 10 pre-synaptic and last 10 post-synaptic neurons
    conn_mask[-10:] = np.arange(20) >= 10
    # Randomly turn off some of the connections enabled in the two groups
    conn_mask *= (np.random.rand(20, 20) < 0.5)

    # Generate random weights for the connections ranging anywhere in the range [0, 2)
    weight = 2 * conn_mask
    conn = pre_synaptic_neurons.connect(post_neurons,
                                        prototype=conn_proto,
                                        connectionMask=conn_mask,
                                        weight=weight)

    # Create connection mask between this first Astrocyte and first 10 neurons in the pre-synaptic neuron group
    astro1_input_mask = np.int_(np.ones((1, pre_neuron_cnt)))
    astro1_input_mask[0][10:] = np.zeros((1, 10))

    # Connect first Astrocyte to first 10 post-synaptic neurons
    astro1_output_mask = np.int_(np.ones((post_neuron_cnt, 1)))
    astro1_output_mask[10:] = np.zeros((10, 1))

    astrocyte1 = combra.Astrocyte(net)
    astrocyte1.connectInputNeurons(pre_synaptic_neurons, pre_neuron_cnt, connectionMask=astro1_input_mask, weight=45)
    astrocyte1.connectOutputNeurons(post_neurons, post_neuron_cnt, connectionMask=astro1_output_mask, weight=5)

    # Create connection mask between second Astrocyte and last 10 neurons in pre-synaptic neuron group
    astro2_input_mask = np.int_(np.ones((1, pre_neuron_cnt)))
    astro2_input_mask[0][:10] = np.zeros((1, 10))

    # Connect second Astrocyte to last 10 post-synaptic neurons
    astro2_output_mask = np.int_(np.ones((post_neuron_cnt, 1)))
    astro2_output_mask[:10] = np.zeros((10, 1))

    astrocyte2 = combra.Astrocyte(net)
    astrocyte2.connectInputNeurons(pre_synaptic_neurons, pre_neuron_cnt, connectionMask=astro2_input_mask, weight=45)
    astrocyte2.connectOutputNeurons(post_neurons, post_neuron_cnt, connectionMask=astro2_output_mask, weight=5)

    # Create probes for plots
    probes = dict()
    probes['post_spikes'] = post_neurons.probe([nx.ProbeParameter.SPIKE])[0]
    probes['astro1_sr_spikes'] = astrocyte1.probe(combra.ASTRO_SPIKE_RECEIVER_PROBE.SPIKE)
    probes['astro1_ip3_voltage'] = astrocyte1.probe(combra.ASTRO_IP3_INTEGRATOR_PROBE.COMPARTMENT_VOLTAGE)
    probes['astro1_sic_voltage'] = astrocyte1.probe(combra.ASTRO_SIC_GENERATOR_PROBE.COMPARTMENT_VOLTAGE)
    probes['astro1_sg_spikes'] = astrocyte1.probe(combra.ASTRO_SPIKE_GENERATOR_PROBE.SPIKE)

    probes['astro2_sr_spikes'] = astrocyte2.probe(combra.ASTRO_SPIKE_RECEIVER_PROBE.SPIKE)
    probes['astro2_ip3_voltage'] = astrocyte2.probe(combra.ASTRO_IP3_INTEGRATOR_PROBE.COMPARTMENT_VOLTAGE)
    probes['astro2_sic_voltage'] = astrocyte2.probe(combra.ASTRO_SIC_GENERATOR_PROBE.COMPARTMENT_VOLTAGE)
    probes['astro2_sg_spikes'] = astrocyte2.probe(combra.ASTRO_SPIKE_GENERATOR_PROBE.SPIKE)

    net.run(sim_time)
    net.disconnect()

    # Plots

    fig = plt.figure(1, figsize=(18, 28))
    ax0 = plt.subplot(7, 1, 1)
    ax0.set_xlim(0, sim_time)
    plotRaster(input_spike_times)
    plt.ylabel('neuron index')
    plt.xlabel('time (ms)')
    plt.title('Presynaptic neurons poisson spikes')

    ax1 = plt.subplot(7, 1, 2)
    ax1.set_xlim(0, sim_time)
    probes['astro1_sr_spikes'].plot()
    probes['astro2_sr_spikes'].plot()
    plt.xlabel('time (ms)')
    plt.title('Astrocyte compartment 1: Spike receiver spikes')

    ax2 = plt.subplot(7, 1, 3)
    ax2.set_xlim(0, sim_time)
    probes['astro1_ip3_voltage'].plot()
    probes['astro2_ip3_voltage'].plot()
    plt.xlabel('time (ms)')
    plt.title('Astrocyte compartment 2: IP3 integrator voltage')

    ax3 = plt.subplot(7, 1, 4)
    ax3.set_xlim(0, sim_time)
    probes['astro1_sic_voltage'].plot()
    probes['astro2_sic_voltage'].plot()
    plt.xlabel('time (ms)')
    plt.title('Astrocyte compartment 3: SIC generator voltage')

    ax4 = plt.subplot(7, 1, 5)
    ax4.set_xlim(0, sim_time)
    probes['astro1_sg_spikes'].plot()
    probes['astro2_sg_spikes'].plot()
    plt.xlabel('time (ms)')
    plt.title('Astrocyte compartment 4: Spike generator spikes')

    ax5 = plt.subplot(7, 1, 6)
    ax5.set_xlim(0, sim_time)
    probes['post_spikes'].plot()
    plt.xlabel('time (ms)')
    plt.title('Post-synaptic neuron spikes')

    plt.tight_layout()
    fileName = "example3_output.svg"
    print("No display available, saving to file " + fileName + ".")
    fig.savefig(fileName)

    combra.FiringRatePlot('Example 3: Post-Synaptic Neuron Firing Rate', './', probes['post_spikes'].data, 'svg')
