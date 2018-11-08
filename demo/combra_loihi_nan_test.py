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


def setupNAN(net: nx.NxNet, sim_time):
    """
    Setup a simple NAN using combra_loihi api

    :param net: NxNet
    :return: probes
    """
    nan = combra.FeedforwardNAN(net, sim_time=sim_time)
    post_probes, astro_probes = nan.probeNAN([nx.ProbeParameter.SPIKE],
                                             [combra.ASTRO_SPIKE_RECEIVER_PROBE.COMPARTMENT_VOLTAGE,
                                              combra.ASTRO_IP3_INTEGRATOR_PROBE.COMPARTMENT_VOLTAGE,
                                              combra.ASTRO_SIC_GENERATOR_PROBE.COMPARTMENT_VOLTAGE,
                                              combra.ASTRO_SPIKE_GENERATOR_PROBE.SPIKE,
                                              combra.ASTRO_IP3_INTEGRATOR_PROBE.SPIKE])
    return nan.poisson_spike, post_probes, astro_probes


def setupAstro(net):
    astro = combra.Astrocyte(net)
    astro_probes = astro.probe([combra.ASTRO_SPIKE_RECEIVER_PROBE.SPIKE,
                                combra.ASTRO_IP3_INTEGRATOR_PROBE.COMPARTMENT_VOLTAGE,
                                combra.ASTRO_SIC_GENERATOR_PROBE.COMPARTMENT_VOLTAGE,
                                combra.ASTRO_SPIKE_GENERATOR_PROBE.SPIKE])
    return astro_probes


if __name__ == '__main__':
    net = nx.NxNet()
    sim_time = 7000
    pre_spikes, post_probes, astro_probes = setupNAN(net, sim_time)
    #astro_probes = setupAstro(net)
    net.run(sim_time)
    net.disconnect()
    """
    start plotting
    """
    fig = plt.figure(1, figsize=(18, 28))
    ax0 = plt.subplot(6, 1, 1)
    ax0.set_xlim(0, sim_time)
    plotRaster(pre_spikes)
    plt.ylabel('neuron index')
    plt.xlabel('time (ms)')
    plt.title('Presynaptic neurons poisson spikes')

    ax1 = plt.subplot(6, 1, 2)
    ax1.set_xlim(0, sim_time)
    astro_probes[0].plot()
    plt.xlabel('time (ms)')
    plt.title('Astrocyte compartment 1: Spike receiver spikes')

    ax2 = plt.subplot(6, 1, 3)
    ax2.set_xlim(0, sim_time)
    astro_probes[1].plot()
    plt.xlabel('time (ms)')
    plt.title('Astrocyte compartment 2: IP3 integrator voltage')

    ax3 = plt.subplot(6, 1, 4)
    ax3.set_xlim(0, sim_time)
    astro_probes[2].plot()
    plt.xlabel('time (ms)')
    plt.title('Astrocyte compartment 3: SIC generator voltage')

    ax4 = plt.subplot(6, 1, 5)
    ax4.set_xlim(0, sim_time)
    astro_probes[3].plot()
    plt.xlabel('time (ms)')
    plt.title('Astrocyte compartment 4: Spike generator spikes')

    ax5 = plt.subplot(6, 1, 6)
    ax5.set_xlim(0, sim_time)
    post_probes[0].plot()
    """
    generator plot
    """
    plt.tight_layout()
    fileName = "combra_loihi_nan_test.png"
    print("No display available, saving to file " + fileName + ".")
    fig.savefig(fileName)
    """
    generate SIC SG firing rate plot
    """
    figure2 = combra.FiringRatePlot('Burst Spike Generator FR plot', '', astro_probes[3].data, 'png', window=30)
