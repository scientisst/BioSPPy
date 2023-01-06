# -*- coding: utf-8 -*-
"""
biosppy.tests.eda
-------------------
This module provides methods to test the eda module functions.
:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# 3rd party
import numpy as np
from matplotlib import pylab as plt

# local
from ..signals.eda import eda_features, eda_events


def test():
    signal = np.loadtxt("examples/eda.txt")  
    
    onsets, peaks, amplitudes, end = eda_events(signal, filt=True, size=1.5, sampling_rate=1000)
    for i in range(len(onsets)):
        plt.figure()
        plt.plot(signal)
        plt.vlines(onsets[i], signal.min(), signal.max(), label="on", color="r")
        plt.vlines(peaks[i], signal.min(), signal.max(), label="peaks", color="b")
        plt.vlines(end[i], signal.min(), signal.max(), label="end", color="c")
        plt.hlines(eda_s[int(onsets[i])]+amplitudes[i], 0, len(signal), label="amplitudes", color="k")
        plt.legend()
        plt.show()

    args = eda_features(signal, filt=True, size=1.5, sampling_rate=1000)

    for i in range(len(onsets)):
        plt.figure()
        plt.plot(signal)
        plt.vlines(args["onsets"][i], signal.min(), signal.max(), label="on", color="r")
        plt.vlines(args["peaks"][i], signal.min(), signal.max(), label="peaks", color="b")
        plt.hlines(eda_s[int(args["onsets"][i])]+args["amps"][i], 0, len(signal), label="amplitudes", color="k")
        
        plt.vlines(args["onsets"][i] + args["rise_ts"][i], signal.min(), signal.max(), label="rise time check", color="c")
        try:
            plt.vlines(args["onsets"][i] + args["half_rec"][i], signal.min(), signal.max(), label="half recovery time", color="pink")
        except Exception as e:
            print(e)
            continue
        try:
            plt.vlines(args["onsets"][i] + args["six_rec"][i], signal.min(), signal.max(), label="six recovery time", color="orange")
        except Exception as e:
            print(e)
            continue
        plt.legend()
        plt.show()





