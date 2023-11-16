# -*- coding: utf-8 -*-
"""
biosppy.signals.acc
-------------------

This module provides methods to process Acceleration (ACC) signals.
Implemented code assumes ACC acquisition from a 3 orthogonal axis reference system.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

Authors:
Afonso Ferreira
Diogo Vieira

"""

# Imports
from __future__ import absolute_import, division, print_function
from six.moves import range
import warnings

# 3rd party
import numpy as np

# local
from .. import plotting, utils
from biosppy.inter_plotting import acc as inter_plotting


def acc(signal=None, sampling_rate=100.0, units=None, path=None, show=True, interactive=False):
    """Process a raw ACC signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw ACC signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    units : str, optional
        The units of the input signal. If specified, the plot will have the
        y-axis labeled with the corresponding units.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show a summary plot.
    interactive : bool, optional
        If True, shows an interactive plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    signal : array
        Raw (unfiltered) ACC signal.
    vm : array
        Vector Magnitude feature of the signal.
    sm : array
        Signal Magnitude feature of the signal.
    freq_features : dict
        Positive Frequency domains (Hz) of the signal.
    amp_features : dict
        Normalized Absolute Amplitudes of the signal.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # extract features
    vm_features, sm_features = time_domain_feature_extractor(signal=signal)
    freq_features, abs_amp_features = frequency_domain_feature_extractor(
        signal=signal, sampling_rate=sampling_rate
    )

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)

    # plot
    if show:
        if interactive:
            inter_plotting.plot_acc(
                ts=ts,  # plotting.plot_acc
                raw=signal,
                vm=vm_features,
                sm=sm_features,
                spectrum={"freq": freq_features, "abs_amp": abs_amp_features},
                path=path,
            )
        else:
            plotting.plot_acc(
                ts=ts,  # plotting.plot_acc
                raw=signal,
                vm=vm_features,
                sm=sm_features,
                units=units,
                path=path,
                show=True,
            )

    # output
    args = (ts, signal, vm_features, sm_features, freq_features, abs_amp_features)
    names = ("ts", "signal", "vm", "sm", "freq", "abs_amp")

    return utils.ReturnTuple(args, names)


def activity_index(signal=None, sampling_rate=100.0, window_1=5, window_2=60):
    """
    Compute the activity index of an ACC signal.

    Follows the method described in [Lin18]_, the activity index is computed as
    follows:
    
    1) Calculate the ACC magnitude if the signal is triaxial
    2) Calculate the standard deviation of the ACC magnitude for each
    'window_1' seconds
    3) The activity index will be the mean standard deviation for each
    'window_2' seconds
    
    Parameters
    ----------
    signal : array
        Raw ACC signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    window_1 : int, float, optional
        Window length (seconds) for the first moving average filter. Default: 5
    window_2 : int, float, optional
        Window length (seconds) for the second moving average filter. Default: 60

    Returns
    -------
    ts : array
        Time axis reference (seconds).
    activity_index : array
        Activity index of the signal.

    References
    ----------
    .. [Lin18] W.-Y. Lin, V. Verma, M.-Y. Lee, C.-S. Lai, Activity
    Monitoring with a Wrist-Worn, Accelerometer-Based Device, Micromachines 9
    (2018) 450

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")
    
    signal = np.array(signal)
    if signal.shape[1] != 3:
        raise TypeError("Input signal must be triaxial.")

    # window size 1
    window_1 = int(sampling_rate * window_1) 
    # window size 2
    window_2 = int(sampling_rate * window_2) 

    # acc magnitude
    acc_magnitude = np.sum(signal**2, axis=1)**0.5
    # activity index is the sum of 12 standard deviations computed for 5s
    # periods
    if window_2 > len(acc_magnitude):
        warnings.warn("'window_2' must be smaller than the signal length. The"
                      "activity index will be computed for the whole signal.")
        window_2 = len(acc_magnitude)
    # last activity index is in (len(acc_magnitude) - len(acc_magnitude) % window_2)
    acc_magnitude = acc_magnitude[:len(acc_magnitude) - len(acc_magnitude) % window_2]

    # segment magnitude in segments with size window_1
    segments = acc_magnitude.reshape(-1, window_1)

    # compute standard deviation for each segment
    activity_index_ = np.std(segments, axis=1)

    # compute activity index (average standard deviation) for each window_2
    activity_index = np.mean(activity_index_.reshape(-1, int(window_2/window_1)), axis=1)

    # timestamps are the end of each window_2 (remove the first one)
    ts = np.arange(0, len(acc_magnitude)+1, int(window_2))[1:]

    # output
    args = (ts, activity_index)
    names = ("ts", "activity_index")

    return utils.ReturnTuple(args, names)


def time_domain_feature_extractor(signal=None):
    """Extracts the vector magnitude and signal magnitude features from an input ACC signal, given the signal itself.

    Parameters
    ----------
    signal : array
        Input ACC signal.

    Returns
    -------
    vm_features : array
        Extracted Vector Magnitude (VM) feature.
    sm_features : array
        Extracted Signal Magnitude (SM) feature.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # get acceleration features
    vm_features = np.zeros(signal.shape[0])
    sm_features = np.zeros(signal.shape[0])

    for i in range(signal.shape[0]):
        vm_features[i] = np.linalg.norm(
            np.array([signal[i][0], signal[i][1], signal[i][2]])
        )
        sm_features[i] = (abs(signal[i][0]) + abs(signal[i][1]) + abs(signal[i][2])) / 3

    return utils.ReturnTuple((vm_features, sm_features), ("vm", "sm"))


def frequency_domain_feature_extractor(signal=None, sampling_rate=100.0):
    """Extracts the FFT from each ACC sub-signal (x, y, z), given the signal itself.

    Parameters
    ----------
    signal : array
        Input ACC signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    freq_features : dict
        Dictionary of positive frequencies (Hz) for all sub-signals.
    amp_features : dict
        Dictionary of Normalized Absolute Amplitudes for all sub-signals.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    freq_features = {}
    amp_features = {}

    # get Normalized FFT for each sub-signal
    for ind, axis in zip(range(signal.shape[1]), ["x", "y", "z"]):
        sub_signal = signal[:, ind]

        n = len(sub_signal)
        k = np.arange(n)
        T = n / sampling_rate
        frq = k / T
        freq_features[axis] = frq[range(n // 2)]

        amp = np.fft.fft(sub_signal) / n
        amp_features[axis] = abs(amp[range(n // 2)])

    return utils.ReturnTuple((freq_features, amp_features), ("freq", "abs_amp"))

