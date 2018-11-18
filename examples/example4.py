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


if __name__ == '__main__':
    net = nx.NxNet()
    sim_time = 7000
    pre_spikes, post_probes, astro_probes = setupNAN(net, sim_time)
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
    fr_data, fr_x = combra.FiringRateComputeGap(astro_probes[3].data)
    ax4.plot(fr_x, fr_data[0, :])
    plt.xlabel('time (ms)')
    plt.title('Astrocyte compartment 4: Spike generator firing rate')

    ax5 = plt.subplot(6, 1, 6)
    ax5.set_xlim(0, sim_time)
    post_probes[0].plot()
    """
    generator plot
    """
    plt.tight_layout()
    fileName = "example4_nan"
    print("No display available, saving to file " + fileName + ".")
    fig.savefig(fileName+'.png', format='png')
