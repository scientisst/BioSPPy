# -*- coding: utf-8 -*-
"""
biosppy.signals.eda
-------------------

This module provides methods to process Electrodermal Activity (EDA)
signals, also known as Galvanic Skin Response (GSR).

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range

# 3rd party
import numpy as np
from scipy import interpolate

# local
from . import tools as st
from .. import plotting, utils


def eda(signal=None, sampling_rate=1000.0, path=None, show=True):
    """Process a raw EDA signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered EDA signal.
    onsets : array
        Indices of SCR pulse onsets.
    peaks : array
        Indices of the SCR peaks.
    amplitudes : array
        SCR pulse amplitudes.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    aux, _, _ = st.filter_signal(
        signal=signal,
        ftype="butter",
        band="lowpass",
        order=4,
        frequency=5,
        sampling_rate=sampling_rate,
    )

    # smooth
    sm_size = int(0.75 * sampling_rate)
    filtered, _ = st.smoother(signal=aux, kernel="boxzen", size=sm_size, mirror=True)

    # get SCR info
    onsets, peaks, amplitudes = emotiphai_eda(signal=filtered,
                                              sampling_rate=sampling_rate,
                                              min_amplitude=0.1,
                                              filt=True,
                                              size=0.9)

    # get time vectors
    length = len(signal)
    t = (length - 1) / sampling_rate
    ts = np.linspace(0, t, length, endpoint=True)

    # get EDR and EDL
    edr_signal = edr(signal=filtered, sampling_rate=sampling_rate)["edr"]
    edl_signal = edl(signal=filtered, sampling_rate=sampling_rate, method="onsets", onsets=onsets)["edl"]

    # plot
    if show:
        plotting.plot_eda(
            ts=ts,
            raw=signal,
            filtered=filtered,
            edr=edr_signal,
            edl=edl_signal,
            onsets=onsets,
            peaks=peaks,
            amplitudes=amplitudes,
            path=path,
            show=True,
        )

    # output
    args = (ts, filtered, edr_signal, edl_signal, onsets, peaks, amplitudes)
    names = ("ts", "filtered", "edr", "edl", "onsets", "peaks", "amplitudes")

    return utils.ReturnTuple(args, names)


def eda_events(signal=None, sampling_rate=1000., method="emotiphai", **kwargs):
    """Returns characteristic EDA events.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : int, float, optional
        Data acquisition sampling rate (Hz).
    method : str, optional
       Method to compute eda events: 'emotiphai', 'kbk' or 'basic'.
    kwargs : dict, optional
        Method parameters.

    Returns
    -------
    onsets : array
        Signal EDR events onsets.
    peaks : array
        Signal EDR events peaks.
    amps : array
        Signal EDR events Amplitudes.
    phasic_rate : array
        Signal EDR events rate in 60s.
    rise_ts : array
        Rise times, i.e. onset-peak time difference.
    half_rec : array
        Half Recovery times, i.e. time between peak and 63% amplitude.
    six_rec : array
        63 % recovery times, i.e. time between peak and 50% amplitude.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure input type
    assert len(signal) > 1, "len signal <1"
    signal = np.array(signal).astype(np.float)

    # compute onsets, peaks and amplitudes
    if method == "emotiphai":
        onsets, peaks, amps = emotiphai_eda(signal=signal, sampling_rate=sampling_rate, **kwargs)

    elif method == "kbk":
        onsets, peaks, amps = kbk_scr(signal=signal, sampling_rate=sampling_rate, **kwargs)

    elif method == "basic":
        onsets, peaks, amps = basic_scr(signal=signal)

    else:
        raise TypeError("Please specify a supported method.")

    # compute phasic rate
    try:
        phasic_rate = sampling_rate * (60. / np.diff(peaks))
    except:
        phasic_rate = None

    # compute rise times
    try:
        rise_ts = peaks - onsets
    except:
        rise_ts = None

    # compute half and 63% recovery times
    half_rec, six_rec = rec_times(signal, onsets, peaks)

    args = (onsets, peaks, amps, phasic_rate, rise_ts, half_rec, six_rec)
    names = ("onsets", "peaks", "amplitudes", "phasic_rate", "rise_ts", "half_rec", "six_rec")

    return utils.ReturnTuple(args, names)


def edr(signal=None, sampling_rate=1000.0):
    """Extracts EDR signal.

    Parameters
    ----------
    signal : array
        Input filtered EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    edr : array
        Electrodermal response (EDR) signal.


    References
    ----------
    .. [KiBK04] K.H. Kim, S.W. Bang, and S.R. Kim, "Emotion recognition
       system using short-term monitoring of physiological signals",
       Med. Biol. Eng. Comput., vol. 42, pp. 419-427, 2004

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # differentiation
    df = np.diff(signal)

    # smooth
    size = int(1.0 * sampling_rate)
    edr_signal, _ = st.smoother(signal=df, kernel="bartlett", size=size, mirror=True)

    # output
    args = (edr_signal,)
    names = ("edr",)

    return utils.ReturnTuple(args, names)


def edl(signal=None, sampling_rate=1000.0, method="onsets", onsets=None, **kwargs):
    """Extracts EDL signal.

    Parameters
    ----------
    signal : array
        Input filtered EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    method: str, optional
        Method to compute the edl signal: "smoother" to compute a smoothing filter; "onsets" to obtain edl by onsets'
        interpolation.
    onsets : array, optional
        List of onsets for the interpolation method.
    kwargs : dict, optional
        window_size : Size of the smoother kernel (seconds).

    Returns
    -------
    edl : array
        Electrodermal level (EDL) signal.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if method == "onsets" and onsets is None:
        raise TypeError("Please specify 'onsets' to use the onset interpolation method.")

    # smooth method
    if method == "smoother":
        window_size = kwargs['window_size'] if 'window_size' in kwargs else 10.0
        size = int(window_size * sampling_rate)
        edl_signal, _ = st.smoother(signal=signal, kernel="bartlett", size=size, mirror=True)

    # interpolation method
    elif method == "onsets":
        # get time vectors
        length = len(signal)
        t = (length - 1) / sampling_rate
        ts = np.linspace(0, t, length, endpoint=True)

        # extract eda
        edl_on = np.hstack((ts[0], ts[onsets], ts[-1]))
        edl_amp = np.hstack((signal[0], signal[onsets], signal[-1]))
        f = interpolate.interp1d(edl_on, edl_amp)
        edl_signal = f(ts)

    else:
        raise TypeError("Please specify a supported method.")

    # output
    args = (edl_signal,)
    names = ("edl",)

    return utils.ReturnTuple(args, names)


def basic_scr(signal=None, sampling_rate=1000.0):
    """
    Basic method to extract Skin Conductivity Responses (SCR) from an
    EDA signal.

    Follows the approach in [Gamb08]_.

    Parameters
    ----------
    signal : array
        Input filterd EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.
    amplitudes : array
        SCR pulse amplitudes.

    References
    ----------
    .. [Gamb08] Hugo Gamboa, "Multi-modal Behavioral Biometrics Based on HCI
       and Electrophysiology", PhD thesis, Instituto Superior T{\'e}cnico, 2008

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # find extrema
    pi, _ = st.find_extrema(signal=signal, mode="max")
    ni, _ = st.find_extrema(signal=signal, mode="min")

    # sanity check
    if len(pi) == 0 or len(ni) == 0:
        raise ValueError("Could not find SCR pulses.")

    # pair vectors
    if ni[0] > pi[0]:
        ni = ni[1:]
    if pi[-1] < ni[-1]:
        pi = pi[:-1]
    if len(pi) > len(ni):
        pi = pi[:-1]

    li = min(len(pi), len(ni))
    i1 = pi[:li]
    i3 = ni[:li]

    # indices
    i0 = np.array((i1 + i3) / 2.0, dtype=int)
    if i0[0] < 0:
        i0[0] = 0

    # amplitude
    a = signal[i0] - signal[i3]

    # output
    args = (i3, i0, a)
    names = ("onsets", "peaks", "amplitudes")

    return utils.ReturnTuple(args, names)


def kbk_scr(signal=None, sampling_rate=1000.0, min_amplitude=0.1):
    """KBK method to extract Skin Conductivity Responses (SCR) from an
    EDA signal.

    Follows the approach by Kim *et al.* [KiBK04]_.

    Parameters
    ----------
    signal : array
        Input filtered EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    min_amplitude : float, optional
        Minimum threshold by which to exclude SCRs.

    Returns
    -------
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.
    amplitudes : array
        SCR pulse amplitudes.

    References
    ----------
    .. [KiBK04] K.H. Kim, S.W. Bang, and S.R. Kim, "Emotion recognition
       system using short-term monitoring of physiological signals",
       Med. Biol. Eng. Comput., vol. 42, pp. 419-427, 2004

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")
    
    # extract edr signal
    df = edr(signal, sampling_rate=sampling_rate)['edr']

    # zero crosses
    (zeros,) = st.zero_cross(signal=df, detrend=False)
    if np.all(df[: zeros[0]] > 0):
        zeros = zeros[1:]
    if np.all(df[zeros[-1] :] > 0):
        zeros = zeros[:-1]

    scrs, amps, ZC, peaks = [], [], [], []
    for i in range(0, len(zeros) - 1, 2):
        scrs += [df[zeros[i] : zeros[i + 1]]]
        ZC += [zeros[i]]
        ZC += [zeros[i + 1]]
        peaks += [zeros[i] + np.argmax(df[zeros[i] : zeros[i + 1]])]
        amps += [signal[peaks[-1]] - signal[ZC[-2]]]

    # exclude SCRs with small amplitude
    thr = min_amplitude * np.max(amps)
    idx = np.where(amps > thr)

    scrs = np.array(scrs, dtype=np.object)[idx]
    amps = np.array(amps)[idx]
    ZC = np.array(ZC)[np.array(idx) * 2]
    peaks = np.array(peaks, dtype=int)[idx]

    onsets = ZC[0].astype(int)

    # output
    args = (onsets, peaks, amps)
    names = ("onsets", "peaks", "amplitudes")

    return utils.ReturnTuple(args, names)


def emotiphai_eda(signal=None, sampling_rate=1000., min_amplitude=0.1, filt=True, size=1.):
    """Returns characteristic EDA events.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    min_amplitude : float, optional
        Minimum threshold by which to exclude SCRs.
    filt: bool, optional
        Whether to filter signal to remove noise and low amplitude events.
    size: float
        Size of the filter in seconds

    Returns
    -------
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.
    amplitudes : array
        SCR pulse amplitudes.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    assert len(signal) > 1, "len signal <1"
    signal = np.array(signal).astype(np.float)

    # smooth
    if filt:
        try:
            if sampling_rate > 1:
                signal, _, _ = st.filter_signal(signal=signal,
                                         ftype='butter',
                                         band='lowpass',
                                         order=4,
                                         frequency=2,
                                         sampling_rate=sampling_rate)
        except Exception as e:
            print(e, "Error filtering EDA")

        # smooth
        try:
            sm_size = int(size * sampling_rate)
            signal, _ = st.smoother(signal=signal,
                                      kernel='boxzen',
                                      size=sm_size,
                                      mirror=True)
        except Exception as e:
            print(e)

    # extract onsets, peaks and amplitudes
    onsets, peaks, amps = [], [], []
    zeros = st.find_extrema(signal=signal, mode='min')[0]  # get zeros
    for z in range(len(zeros)):
        if z == len(zeros) - 1:  # last zero
            s = signal[zeros[z]:]  # signal amplitude between event
        else:
            s = signal[zeros[z]:zeros[z + 1]]  # signal amplitude between event
            
        pk = st.find_extrema(signal=s, mode='max')[0]  # get pk between events
        for p in pk:
            if (s[p] - s[0]) > min_amplitude:  # only count events with high amplitude
                peaks += [zeros[z] + p]
                onsets += [zeros[z]]
                amps += [s[p] - s[0]]

    # convert to array
    onsets, peaks, amps = np.array(onsets), np.array(peaks), np.array(amps)

    # output
    args = (onsets, peaks, amps)
    names = ("onsets", "peaks", "amplitudes")

    return utils.ReturnTuple(args, names)


def rec_times(signal=None, onsets=None, peaks=None):
    """Returns EDA recovery times.

    Parameters
    ----------
    signal : array
        Input signal.
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.

    Returns
    -------
    half_rec : list
        Half Recovery times, i.e. time between peak and 50% amplitude.
    six_rec : list
        63 % recovery times, i.e. time between peak and 63% amplitude.

    """
    assert len(signal) > 1, "len signal <1"
    peaks = np.array(peaks).astype(int) 
    onsets = np.array(onsets).astype(int) 

    a = np.array(signal[peaks[:]] - signal[onsets[:]])
    li = min(len(onsets), len(peaks))

    half_rec, hlf_rec_ts = [], []
    six_rec, six_rec_ts = [], []
    for i in range(li):  # iterate over onset
        half_rec_amp = 0.5 * a[i] + signal[onsets][i]
        six_rec_amp = 0.37 * a[i] + signal[onsets][i]
        try:
            wind = np.array(signal[peaks[i]:onsets[i + 1]])
        except:
            wind = np.array(signal[peaks[i]:])
        half_rec_idx = np.argwhere(wind <= half_rec_amp)
        six_rec_idx = np.argwhere(wind <= six_rec_amp)
        
        if len(half_rec_idx) > 0:
            half_rec += [half_rec_idx[0][0] + peaks[i] - onsets[i]]
            hlf_rec_ts += [half_rec_idx[0][0] + peaks[i]]
        else:
            half_rec += [None]
            hlf_rec_ts += [None]

        if len(six_rec_idx) > 0:
            six_rec += [six_rec_idx[0][0] + peaks[i] - onsets[i]]
            six_rec_ts += [six_rec_idx[0][0] + peaks[i]]
        else:
            six_rec += [None]
            six_rec_ts += [None]

    args, names = [], []
    names += ["half_rec", "six_rec"]
    args += [half_rec, six_rec]
    args = np.nan_to_num(args)
    return utils.ReturnTuple(tuple(args), tuple(names))
