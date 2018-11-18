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

from enum import IntEnum


class ASTRO_SPIKE_RECEIVER_PROBE(IntEnum):
    COMPARTMENT_CURRENT = 1
    COMPARTMENT_VOLTAGE = 2
    SPIKE = 3


class ASTRO_IP3_INTEGRATOR_PROBE(IntEnum):
    COMPARTMENT_CURRENT = 4
    COMPARTMENT_VOLTAGE = 5
    SPIKE = 6


class ASTRO_SIC_GENERATOR_PROBE(IntEnum):
    COMPARTMENT_CURRENT = 7
    COMPARTMENT_VOLTAGE = 8


class ASTRO_SPIKE_GENERATOR_PROBE(IntEnum):
    COMPARTMENT_CURRENT = 9
    COMPARTMENT_VOLTAGE = 10
    SPIKE = 11
