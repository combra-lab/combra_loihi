import nxsdk.api.n2a as nx
from combra_loihi.astro.astrocyte import Astrocyte
import numpy as np


class FeedforwardNAN:
    def __init__(self,
                 net: nx.NxNet,
                 pre_num=50,
                 post_num=50,
                 pre_fr=10,
                 pre_post_w=10,
                 pre_post_conn_p=0.1,
                 post_vth=100,
                 post_cdecay=int(1/10*2**12),
                 post_vdecay=int(1/4*2**12),
                 sim_time=30000):
        """

        :param net: NxNet
        :param pre_num: number of presynaptic neurons
        :param post_num: number of postsynaptic neurons
        :param pre_fr: presynaptic neuron mean firing rate
        :param pre_post_w: weight between presynaptic neuron and postsynaptic neuron
        :param pre_post_conn_p: connection density
        :param post_vth: postsynaptic neuron vth
        :param post_cdecay: postsynaptic neuron current decay
        :param post_vdecay: postsynaptic neuron voltage decay
        :param sim_time: simulation time in ms
        """
        assert isinstance(net, nx.NxNet)
        assert isinstance(pre_num, int)
        assert isinstance(post_num, int)
        assert isinstance(pre_fr, int)
        assert isinstance(pre_post_w, int)
        assert isinstance(pre_post_conn_p, float)
        assert isinstance(post_vth, int)
        assert isinstance(post_cdecay, int)
        assert isinstance(post_vdecay, int)
        assert isinstance(sim_time, int)
        self.pre_num = pre_num
        self.post_num = post_num
        self.pre_fr = pre_fr
        self.pre_post_w = pre_post_w
        self.pre_post_conn_p = pre_post_conn_p
        self.post_vth = post_vth
        self.post_cdecay = post_cdecay
        self.post_vdecay = post_vdecay
        self.sim_time = sim_time
        """
        Define network
        """
        self.net = net
        self.poisson_spike, self.pre_2_post_conn, self.post_neurons, self.astrocyte = self.__core()

    def __core(self):
        """
        Private function for setup feedforward nan

        :return: poisson_spikes: list
        :return: pre_2_post_conn: nx.Connection
        :return: post_neurons: nx.CompartmentGroup
        :return: astrocyte: combra.Astrocyte
        """
        """
        define spike generator as presynaptic neurons
        """
        pre_neurons = self.net.createSpikeGenProcess(self.pre_num)
        random_spikes = np.random.rand(self.pre_num, self.sim_time) < (self.pre_fr / 1000.)
        poisson_spikes = [np.where(random_spikes[num, :])[0].tolist() for num in range(self.pre_num)]
        # add spikes to spike generator
        pre_neurons.addSpikes(
            spikeInputPortNodeIds=[num for num in range(self.pre_num)],
            spikeTimes=poisson_spikes
        )
        """
        define post synaptic neurons
        """
        post_neurons_prototype = nx.CompartmentPrototype(
            vThMant=self.post_vth,
            compartmentCurrentDecay=self.post_cdecay,
            compartmentVoltageDecay=self.post_vdecay,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE
        )
        post_neurons = self.net.createCompartmentGroup(
            size=self.post_num,
            prototype=post_neurons_prototype
        )
        """
        define astrocyte
        """
        astrocyte = Astrocyte(self.net)
        """
        define connection between presynaptic neurons and postsynaptic neurons
        """
        pre_2_post_conn_prototype = nx.ConnectionPrototype()
        mask = np.int_(np.random.rand(self.post_num, self.pre_num) < self.pre_post_conn_p)
        weight = self.pre_post_w * mask
        pre_2_post_conn = pre_neurons.connect(
            post_neurons,
            prototype=pre_2_post_conn_prototype,
            connectionMask=mask,
            weight=weight
        )
        """
        define connection between neurons and astrocyte
        """
        astrocyte.connectInputNeurons(pre_neurons, self.pre_num)
        astrocyte.connectOutputNeurons(post_neurons, self.post_num)
        """
        return
        """
        return poisson_spikes, pre_2_post_conn, post_neurons, astrocyte

    def probeNAN(self, postConditions, astroConditions):
        """
        create probes for nan networks

        :param postConditions: int for single probe, list for list of probes
        :param astroConditions: int for single probe, list for list of probes
        :return: postProbes
        :return: astroProbes
        """
        postProbes = self.post_neurons.probe(postConditions)
        astroProbes = self.astrocyte.probe(astroConditions)
        return postProbes, astroProbes

