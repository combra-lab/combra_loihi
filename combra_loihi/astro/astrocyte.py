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

import nxsdk.api.n2a as nx
from nxsdk.arch.n2a.net.process.basicspikegen import BasicSpikeGen
from combra_loihi.astro.astrocyte_base import AstrocytePrototypeBase
import numpy as np


class Astrocyte(AstrocytePrototypeBase):
    def __init__(self,
                 net: nx.NxNet,
                 ip3_sensitivity=None,
                 sic_amplitude=None,
                 sic_window=None,
                 srVThMant=100,
                 srCurrentDecay=int(1 / 10 * 2 ** 12),
                 srVoltageDecay=int(1 / 4 * 2 ** 12),
                 srActivityImpulse=0,
                 srActivityTimeConstant=0,
                 srMinActivity=0,
                 srMaxActivity=127,
                 srHomeostasisGain=0,
                 srEnableHomeostasis=0,
                 ip3VThMant=15000,
                 ip3CurrentDecay=int(2 ** 12),
                 ip3VoltageDecay=1,
                 sicCurrentDecay=int(1 / 100 * 2 ** 12),
                 sicVoltageDecay=int(1 / 100 * 2 ** 12),
                 sgVThMant=5000,
                 sgCurrentDecay=int(1 / 10 * 2 ** 12),
                 sgVoltageDecay=int(1 / 100 * 2 ** 12),
                 sr2ip3Weight=20,
                 ip32sicWeight=20,
                 DEBUG=False):
        super().__init__(net,
                         ip3_sensitivity,
                         sic_amplitude,
                         sic_window,
                         srVThMant,
                         srCurrentDecay,
                         srVoltageDecay,
                         srActivityImpulse,
                         srActivityTimeConstant,
                         srMinActivity,
                         srMaxActivity,
                         srHomeostasisGain,
                         srEnableHomeostasis,
                         ip3VThMant,
                         ip3CurrentDecay,
                         ip3VoltageDecay,
                         sicCurrentDecay,
                         sicVoltageDecay,
                         sgVThMant,
                         sgCurrentDecay,
                         sgVoltageDecay,
                         sr2ip3Weight,
                         ip32sicWeight,
                         DEBUG)

        # declare internal properties
        # ---------------------------------------------------
        # astrocyte compartment and connection list
        self.astrocyte_setup = self.__core()
        self.astrocyte_input_conn = None
        self.astrocyte_output_conn = None
        # ---------------------------------------------------

    def __core(self):
        """
        Private function for core function of Astrocyte computation
        Point Astrocyte is consists 4 compartments
        spike_receiver: spiking compartment receive all spikes from presynaptic neurons
        ip3_integrator: slow spiking compartment integrate spikes from spike_receiver
        sic_generator: non-spike compartment generate voltage sic from ip3 spike
        spike_generator: spiking compartment

        :return: spike_receiver: nx.Compartment
        :return: sr_2_ip3_conn: nx.Connection
        :return: ip3_integrator: nx.Compartment
        :return: ip3_2_sic_conn: nx.Connection
        :return: sic_generator: nx.Compartment
        :return: spike_generator: nx.CompartmentGroup
        """
        spike_receiver_prototype = nx.CompartmentPrototype(
            vThMant=self.srVThMant,
            compartmentCurrentDecay=self.srCurrentDecay,
            compartmentVoltageDecay=self.srVoltageDecay,
            activityImpulse=self.srActivityImpulse,
            activityTimeConstant=self.srActivityTimeConstant,
            enableHomeostasis=self.srEnableHomeostasis,
            maxActivity=self.srMinActivity,
            minActivity=self.srMaxActivity,
            homeostasisGain=self.srHomeostasisGain,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE
        )
        ip3_integrator_prototype = nx.CompartmentPrototype(
            vThMant=self.ip3VThMant,
            compartmentCurrentDecay=self.ip3CurrentDecay,
            compartmentVoltageDecay=self.ip3VoltageDecay,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE
        )
        sic_generator_prototype = nx.CompartmentPrototype(
            compartmentCurrentDecay=self.sicCurrentDecay,
            compartmentVoltageDecay=self.sicVoltageDecay,
            thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.NO_SPIKE_AND_PASS_V_LG_VTH_TO_PARENT,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            stackOut=nx.COMPARTMENT_OUTPUT_MODE.PUSH
        )
        spike_generator_prototype = nx.CompartmentPrototype(
            vThMant=self.sgVThMant,
            compartmentCurrentDecay=self.sgCurrentDecay,
            compartmentVoltageDecay=self.sgVoltageDecay,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            compartmentJoinOperation=nx.COMPARTMENT_JOIN_OPERATION.ADD,
            stackIn=nx.COMPARTMENT_INPUT_MODE.POP_A
        )
        sr_2_ip3_conn_prototype = nx.ConnectionPrototype(signMode=2, numWeightBits=8, weight=self.sr2ip3Weight)
        ip3_2_sic_conn_prototype = nx.ConnectionPrototype(signMode=2, numWeightBits=8, weight=self.ip32sicWeight)
        """
        Astrocyte model part 1: simulate IP3 integration
        """
        spike_receiver = self.net.createCompartment(prototype=spike_receiver_prototype)
        ip3_integrator = self.net.createCompartment(prototype=ip3_integrator_prototype)
        sr_2_ip3_conn = spike_receiver.connect(ip3_integrator, prototype=sr_2_ip3_conn_prototype)
        """
        Astrocyte model part 2: simulate SIC
        """
        sic_generator = self.net.createCompartment(prototype=sic_generator_prototype)
        spike_generator_tmp = self.net.createCompartment(prototype=spike_generator_prototype)
        spike_generator = self.net.createCompartmentGroup()
        spike_generator.addCompartments([spike_generator_tmp])
        ip3_2_sic_conn = ip3_integrator.connect(sic_generator, prototype=ip3_2_sic_conn_prototype)
        """
        return
        """
        return [spike_receiver, sr_2_ip3_conn, ip3_integrator, ip3_2_sic_conn, sic_generator, spike_generator]

    def connectInputNeurons(self, inputs, num, connectionMask=1, weight=10):
        """
        connection Presynaptic neurons with astrocyte

        :param inputs: CompartmentGroup
        :param num: input number
        :param connectionMask: int for full connection, numpy for connection
        :param weight: int for full connection, numpy for connection
        :return:
        """
        assert (isinstance(inputs, nx.CompartmentGroup) or isinstance(inputs, BasicSpikeGen))
        mask = connectionMask
        w = weight
        if isinstance(mask, int):
            mask = np.int_(np.ones((1, num)))
        if isinstance(w, int):
            w = np.int_(np.ones((1, num))) * w
        assert isinstance(mask, np.ndarray)
        assert isinstance(w, np.ndarray)
        assert (mask.shape[1] == num)
        assert (w.shape[1] == num)
        """
        Create connection
        """
        input_conn_prototype = nx.ConnectionPrototype(numWeightBits=8, signMode=2)
        self.astrocyte_input_conn = inputs.connect(
            self.astrocyte_setup[0],
            prototype=input_conn_prototype,
            connectionMask=mask,
            weight=w
        )

    def connectOutputNeurons(self, outputs, num, connectionMask=1, weight=30):
        """
        connection Postsynaptic neurons with astrocyte

        :param outputs: CompartmentGroup
        :param num: output number
        :param connectionMask: int for full connection, numpy for connection
        :param weight: int for full connection, numpy for connection
        :return:
        """
        assert isinstance(outputs, nx.CompartmentGroup)
        mask = connectionMask
        w = weight
        if isinstance(mask, int):
            mask = np.int_(np.ones((num, 1)))
        if isinstance(w, int):
            w = np.int_(np.ones((num, 1))) * w
        assert isinstance(mask, np.ndarray)
        assert isinstance(w, np.ndarray)
        assert (mask.shape[0] == num)
        assert (w.shape[0] == num)
        """
        Create connection
        """
        output_conn_prototype = nx.ConnectionPrototype(numWeightBits=8, signMode=2)
        self.astrocyte_output_conn = self.astrocyte_setup[-1].connect(
            outputs,
            prototype=output_conn_prototype,
            connectionMask=mask,
            weight=w
        )

    def probe(self, probeConditions):
        """
        create probes for astrocyte compartments

        :param probeConditions: int for single plot, list for list of probes
        :return: probe objects
        """
        if isinstance(probeConditions, int):
            assert (probeConditions > 0 or probeConditions < 12)
            """
            compute what probe is needed
            """
            if probeConditions in [1, 2, 3]:
                probe_compartment = self.astrocyte_setup[0]
                probe_choice = probeConditions
            elif probeConditions in [4, 5, 6]:
                probe_compartment = self.astrocyte_setup[2]
                probe_choice = probeConditions - 3
            elif probeConditions in [7, 8]:
                probe_compartment = self.astrocyte_setup[4]
                probe_choice = probeConditions - 6
            else:
                probe_compartment = self.astrocyte_setup[5]
                probe_choice = probeConditions - 8
            """
            generate probe
            """
            if probe_choice == 1:
                astrocyte_probe = probe_compartment.probe([nx.ProbeParameter.COMPARTMENT_CURRENT])
            elif probe_choice == 2:
                astrocyte_probe = probe_compartment.probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE])
            else:
                astrocyte_probe = probe_compartment.probe([nx.ProbeParameter.SPIKE])
            return astrocyte_probe[0]
        else:
            assert isinstance(probeConditions, list)
            """
            generate probe list
            """
            astrocyte_probe = []
            for condition in probeConditions:
                astrocyte_probe.append(self.probe(condition))
            return astrocyte_probe
