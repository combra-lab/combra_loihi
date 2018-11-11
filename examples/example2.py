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
    sim_time = 6000

    pre_neuron_cnt = 10
    post_neuron_cnt = 10

    # Create pre-synaptic neuron (spike generator)
    pre_synaptic_neurons = net.createSpikeGenProcess(pre_neuron_cnt)
    input_spike_times = gen_rand_spikes(pre_neuron_cnt, sim_time, 10)
    pre_synaptic_neurons.addSpikes(
        spikeInputPortNodeIds=[num for num in range(pre_neuron_cnt)],
        spikeTimes=input_spike_times)

    # Create post-synaptic neuron
    post_neuron_proto = nx.CompartmentPrototype(
        vThMant=10,
        compartmentCurrentDecay=int(1/10*2**12),
        compartmentVoltageDecay=int(1/4*2**12),
        functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)
    post_neurons = net.createCompartmentGroup(
        size=post_neuron_cnt, prototype=post_neuron_proto)

    # Create randomly weighted connections from the pre to post-synaptic neurons

    conn_proto = nx.ConnectionPrototype()

    # Create a mask for connections from each input to all the post-synaptic neurons
    conn_mask = np.ones((10, 10))

    # Generate random weights ranging anywhere [0, 5)
    weight = np.random.rand(10, 10) * 5
    weight = weight * conn_mask

    conn = pre_synaptic_neurons.connect(post_neurons,
                                        prototype=conn_proto,
                                        connectionMask=conn_mask,
                                        weight=weight)

    # Create Astrocyte and establish connections
    astrocyte = combra.Astrocyte(net)
    astrocyte.connectInputNeurons(pre_synaptic_neurons, pre_neuron_cnt, weight=45)
    astrocyte.connectOutputNeurons(post_neurons, post_neuron_cnt, weight=5)

    # Create probes for plots
    probes = dict()
    probes['post_spikes'] = post_neurons.probe([nx.ProbeParameter.SPIKE])[0]
    probes['astro_sr_spikes'] = astrocyte.probe(combra.ASTRO_SPIKE_RECEIVER_PROBE.SPIKE)
    probes['astro_ip3_voltage'] = astrocyte.probe(combra.ASTRO_IP3_INTEGRATOR_PROBE.COMPARTMENT_VOLTAGE)
    probes['astro_sic_voltage'] = astrocyte.probe(combra.ASTRO_SIC_GENERATOR_PROBE.COMPARTMENT_VOLTAGE)
    probes['astro_sg_spikes'] = astrocyte.probe(combra.ASTRO_SPIKE_GENERATOR_PROBE.SPIKE)

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
    probes['astro_sr_spikes'].plot()
    plt.xlabel('time (ms)')
    plt.title('Astrocyte compartment 1: Spike receiver spikes')

    ax2 = plt.subplot(7, 1, 3)
    ax2.set_xlim(0, sim_time)
    probes['astro_ip3_voltage'].plot()
    plt.xlabel('time (ms)')
    plt.title('Astrocyte compartment 2: IP3 integrator voltage')

    ax3 = plt.subplot(7, 1, 4)
    ax3.set_xlim(0, sim_time)
    probes['astro_sic_voltage'].plot()
    plt.xlabel('time (ms)')
    plt.title('Astrocyte compartment 3: SIC generator voltage')

    ax4 = plt.subplot(7, 1, 5)
    ax4.set_xlim(0, sim_time)
    probes['astro_sg_spikes'].plot()
    plt.xlabel('time (ms)')
    plt.title('Astrocyte compartment 4: Spike generator spikes')

    ax5 = plt.subplot(7, 1, 6)
    ax5.set_xlim(0, sim_time)
    probes['post_spikes'].plot()
    plt.xlabel('time (ms)')
    plt.title('Post-synaptic neuron spikes')

    plt.tight_layout()
    fileName = "example2_output.svg"
    print("No display available, saving to file " + fileName + ".")
    fig.savefig(fileName)

    combra.FiringRatePlot('Example 2: Post-Synaptic Neuron Firing Rate', './', probes['post_spikes'].data, 'svg')