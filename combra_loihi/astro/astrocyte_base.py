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

"""
This module contains the base class for the Astrocyte class.
"""
import numpy as np
import nxsdk.api.n2a as nx


class AstrocyteInterfaceBase():
    # --------------------------------------
    # Validators
    pass


class AstrocytePrototypeBase(AstrocyteInterfaceBase):
    def __init__(self,
                 net: nx.NxNet,
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
                 ip32sicWeight):
        """
        Initialize the parameters of the astrocyte model.

        :param ip3_sensitivity: Spike time gap of ip3 integrator in ms
        :param sic_amplitude: Max firing rate of SIC spike generator in Hz
        :param sic_window: Firing window of SIC spike generator in ms
        """
        # Loihi net
        self.net = net

        # Astrocyte Core Properties
        # ---------------------------------------------------
        # Spike Receiver Properties
        self.srVThMant = srVThMant
        self.srCurrentDecay = srCurrentDecay
        self.srVoltageDecay = srVoltageDecay
        self.srActivityImpulse = srActivityImpulse
        self.srActivityTimeConstant = srActivityTimeConstant
        self.srMinActivity = srMinActivity
        self.srMaxActivity = srMaxActivity
        self.srHomeostasisGain = srHomeostasisGain
        self.srEnableHomeostasis = srEnableHomeostasis
        # IP3 unit Properties
        self.ip3VThMant = ip3VThMant
        self.ip3CurrentDecay = ip3CurrentDecay
        self.ip3VoltageDecay = ip3VoltageDecay
        # SIC Properties
        self.sicCurrentDecay = sicCurrentDecay
        self.sicVoltageDecay = sicVoltageDecay
        # Spike Generator Properties
        self.sgVThMant = sgVThMant
        self.sgCurrentDecay = sgCurrentDecay
        self.sgVoltageDecay = sgVoltageDecay
        # Spike Receiver to IP3 unit connection weight
        self.sr2ip3Weight = sr2ip3Weight
        self.ip32sicWeight = ip32sicWeight
        # ---------------------------------------------------

        # Smart Setup Properties
        # ---------------------------------------------------
        self.ip3Sensitivity = ip3_sensitivity
        self.sicAmplitude = sic_amplitude
        self.sicWindow = sic_window
        # ---------------------------------------------------

    @property
    def ip3Sensitivity(self):
        """
        read ip3 sensitivity time of ip3 integrator spikes in ms

        :return:
        """
        return self._ip3Sensitivity

    @property
    def sicAmplitude(self):
        """
        read sic amplitude of max sic spike generator firing rate in hz

        :return:
        """
        return self._sicAmplitude

    @property
    def sicWindow(self):
        """
        read sic window of sic spike generator spike window in ms

        :return:
        """
        return self._sicWindow

    @ip3Sensitivity.setter
    def ip3Sensitivity(self, val):
        """
        Set ip3 sensitivity and transform into Loihi Parameters

        :param val: ip3 spike time in ms
        :return:
        """
        self._ip3Sensitivity = val

    @sicAmplitude.setter
    def sicAmplitude(self, val):
        """
        Set sic amplitude and transform into Loihi Parameters

        :param val: sic firing rate in hz
        :return:
        """
        self._sicAmplitude = val

    @sicWindow.setter
    def sicWindow(self, val):
        """
        Set sic window and transform into Loihi Parameters

        :param val: sic firing window in ms
        :return:
        """
        self._sicWindow = val
