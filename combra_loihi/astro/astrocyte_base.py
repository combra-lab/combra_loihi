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
import os
import numpy as np
import nxsdk.api.n2a as nx


class AstrocyteInterfaceBase():
    # --------------------------------------
    # Validators
    @staticmethod
    def _validate_sic_window(val):
        assert 0 <= val <= 608, \
            "Illegal SIC window size = " + str(val) + " ms. " + \
            "Must be an integer >= 0 and <= 608. For a value greater than 608 ms, please configure manually."

    @staticmethod
    def _validate_sic_firing_rate(val):
        assert 0 <= val <= 356, \
            "Illegal SIC maximum firing rate = " + str(val) + " Hz. " + \
            "Must be an integer >= 0 and <= 356. For a value greater than 356 Hz., please configure manually."

    @staticmethod
    def _validate_ip3_sensitivity(val):
        if not isinstance(val, int) or val < 1 or val > 100:
            raise ValueError("IP3 Sensitivity value must be an integer between 1 and 100 (inclusive)")


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
                 ip32sicWeight,
                 DEBUG=False):
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
        if sic_window is not None and sic_amplitude is not None:
            if DEBUG:
                print("DEBUG: Configuring based on provided window size and maximum firing rate")
            self._validate_sic_window(sic_window)
            self._validate_sic_firing_rate(sic_amplitude)
            self.ip32sicWeight, self.sicCurrentDecay = AstrocytePrototypeBase._calculate_sic_props(sic_amplitude,
                                                                                                   sic_window)
            self.sicCurrentDecay = int(self.sicCurrentDecay * 2 ** 12)
            self._sicWindow = sic_window
            self._sicAmplitude = sic_amplitude

        if ip3_sensitivity is not None:
            if DEBUG:
                print("DEBUG: Configuring based on provided IP3 Sensitivity level")
                self.ip3Sensitivity = ip3_sensitivity

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

        self._validate_ip3_sensitivity(val)
        self._ip3Sensitivity = val
        self.sr2ip3Weight = self._ip3Sensitivity

    @sicAmplitude.setter
    def sicAmplitude(self, val):
        """
        Set sic amplitude and transform into Loihi Parameters

        :param val: sic firing rate in hz
        :return:
        """
        self._validate_sic_firing_rate(val)
        self._sicAmplitude = val

        self.ip32sicWeight, self.sicCurrentDecay = AstrocytePrototypeBase._calculate_sic_props(self._sicAmplitude,
                                                                                               self._sicWindow)
        self.sicCurrentDecay = int(self.sicCurrentDecay * 2 ** 12)

    @sicWindow.setter
    def sicWindow(self, val):
        """
        Set sic window and transform into Loihi Parameters

        :param val: sic firing window in ms
        :return:
        """
        self._validate_sic_window(val)
        self._sicWindow = val

        self.ip32sicWeight, self.sicCurrentDecay = AstrocytePrototypeBase._calculate_sic_props(self._sicAmplitude,
                                                                                               self._sicWindow)
        self.sicCurrentDecay = int(self.sicCurrentDecay * 2 ** 12)

    @staticmethod
    def _calculate_sic_props(firing_rate, window_size):
        """
        Calculate the optimal values to achieve closest specifications to those provided for the SIC.

        :param firing_rate:
        :param window_size:
        :return: ip32sicWeight, sicCurrentDecay
        """
        configs = np.load(os.path.join(os.path.dirname(__file__), "sic_data_table.npy"))

        optimal_config = configs[15]
        min_diff = AstrocytePrototypeBase._calc_diff(optimal_config[2], optimal_config[3], firing_rate, window_size)
        for config in configs:
            cost = AstrocytePrototypeBase._calc_diff(config[2], config[3], firing_rate, window_size)
            if min_diff > cost:
                min_diff = cost
                optimal_config = config
        return optimal_config[0], optimal_config[1]

    @staticmethod
    def _calc_diff(config_fr, config_ws, firing_rate, window_size):
        return np.power(config_fr - firing_rate, 2) + np.power(config_ws - window_size, 2)
