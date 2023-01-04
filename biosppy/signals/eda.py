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


def edr(signal=None, sampling_rate=1000.0):
    """
    Extracts EDR signal.

    Parameters
    ----------
    signal : array
        Input filterd EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    edr : array
        Electrodermal response (EDR) signal.

    """
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # differentiation
    df = np.diff(signal)

    # smooth
    size = int(1.0 * sampling_rate)
    edr, _ = st.smoother(signal=df, kernel="bartlett", size=size, mirror=True)

    # output
    args = (edr, )
    names = ("edr", )

    return utils.ReturnTuple(args, names)


def edl(signal=None, sampling_rate=1000.0, method="onsets"):
    """
    Extracts EDL signal.

    Parameters
    ----------
    signal : array
        Input filterd EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    method: string
        Method to compute the edl signal: "smoother" to compute a smoothing filter; "onsets" to obtain edl by onsets interpolation. 

    Returns
    -------
    edl : array
        Electrodermal level (EDL) signal.

    """
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # smooth
    if method == "smoother":
        size = int(10.0 * sampling_rate)
        edl, _ = tools.smoother(signal=signal, kernel="bartlett", size=size, mirror=True)
    else:
        # get time vectors
        length = len(signal)
        T = (length - 1) / sampling_rate
        ts = np.linspace(0, T, length, endpoint=True)

        onsets, peaks, amps, _ = eda_events(signal, filt=True, size=0.9, sampling_rate=1000)
        edl_on = np.hstack((ts[0], ts[onsets], ts[-1]))
        edl_amp = np.hstack((signal[0], signal[onsets], signal[-1]))
        f = interpolate.interp1d(edl_on, edl_amp)
        edl = f(ts)
    # output
    args = (edl, )
    names = ("edl", )

    return utils.ReturnTuple(args, names)


def eda(signal=None, sampling_rate=1000.0, path=None, show=True, min_amplitude=0.1):
    """
    Process a raw EDA signal and extract relevant signal features using
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
    min_amplitude : float, optional
        Minimum treshold by which to exclude SCRs.

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
    onsets, peaks, amplitudes, _ = eda_events(signal, filt=True, size=0.9, sampling_rate=1000)
     
    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)

    _edr = edr(filtered, sampling_rate=sampling_rate)["edr"]
    _edl = edl(filtered, sampling_rate=sampling_rate)["edl"]

    # plot
    if show:
        plotting.plot_eda_(
            ts=ts,
            raw=signal,
            filtered=filtered,
            edr=_edr,
            edl=_edl,
            onsets=onsets,
            peaks=peaks,
            amplitudes=amplitudes,
            path=path,
            show=True,
        )

    # output
    args = (ts, filtered, _edr, _edl, onsets, peaks, amplitudes)
    names = ("ts", "filtered", "edr", "edl", "onsets", "peaks", "amplitudes")

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
    """
    KBK method to extract Skin Conductivity Responses (SCR) from an
    EDA signal.

    Follows the approach by Kim *et al.* [KiBK04]_.

    Parameters
    ----------
    signal : array
        Input filterd EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    min_amplitude : float, optional
        Minimum treshold by which to exclude SCRs.

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

    scrs, amps, ZC, pks = [], [], [], []
    for i in range(0, len(zeros) - 1, 2):
        scrs += [df[zeros[i] : zeros[i + 1]]]
        ZC += [zeros[i]]
        ZC += [zeros[i + 1]]
        pks += [zeros[i] + np.argmax(df[zeros[i] : zeros[i + 1]])]
        amps += [signal[pks[-1]] - signal[ZC[-2]]]

    # exclude SCRs with small amplitude
    thr = min_amplitude * np.max(amps)
    idx = np.where(amps > thr)

    scrs = np.array(scrs, dtype=np.object)[idx]
    amps = np.array(amps)[idx]
    ZC = np.array(ZC)[np.array(idx) * 2]
    pks = np.array(pks, dtype=int)[idx]

    onsets = ZC[0].astype(int)

    # output
    args = (onsets, pks, amps)
    names = ("onsets", "peaks", "amplitudes")

    return utils.ReturnTuple(args, names)


def eda_events(signal, min_amplitude=0.1, filt=True, size=1., sampling_rate=1000.):
    """ 
    Returns characteristic EDA events.

    Parameters
    ----------
    signal : array
        Input signal.
    min_amplitude : float, optional
        Minimum threshold by which to exclude SCRs.
    filt: bool
        If to filter signal to remove noise and low amplitude events.
    size: float
        Size of the filter in seconds
    sampling_rate: float
        Data acquisition sampling rate.

    Returns
    -------
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.
    amplitudes : array
        SCR pulse amplitudes.
    end : array
        Indices of the SCR end.
    """
    
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
        except Exception as e: print(e)


    amps, onsets, peaks, end = [], [], [], []
    zeros = st.find_extrema(signal=signal, mode='min')[0]  # get zeros
    for z in range(len(zeros)):
        if z == len(zeros) -1:  # last zero
            s = signal[zeros[z]:]  # signal amplitude between event
        else:
            s = signal[zeros[z]:zeros[z + 1]]  # signal amplitude between event
            
        pk = st.find_extrema(signal=s, mode='max')[0]  # get pk between events
        for p in pk:
            if (s[p] - s[0]) > min_amplitude:  # only count events with high amplitude
                peaks += [zeros[z] + p]
                onsets += [zeros[z]]
                amps += [s[p] - s[0]]
                if z == len(zeros) -1:  # last zero
                    end += [len(signal)]                    
                else:
                    end += [zeros[z + 1]]

    args, names = [], []
    names += ["onsets", "peaks", "amplitudes", "end"]
    args += [np.array(onsets), np.array(peaks), np.array(amps), np.array(end)]

    return utils.ReturnTuple(tuple(args), tuple(names))


def rec_times(signal, onsets, peaks):
    """ 
    Returns EDA recovery times.

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


def eda_features(signal=None, min_amplitude=0.08, filt=True, size= 1.5, sampling_rate=1000.):
    """
    Compute EDA characteristic metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.
    min_amplitude : float, optional
        Minimum threshold by which to exclude SCRs.
    filt: bool
        If to filter signal to remove noise and low amplitude events.
    size: float
        Size of the filter in seconds
    sampling_rate : float
        Sampling frequency.

    Returns
    -------
    onsets : list
        Signal EDR events onsets.
    peaks : list
        Signal EDR events peaks.
    amps : list
        Signal EDR events Amplitudes.
    phasic_rate : list
        Signal EDR events rate in 60s.
    rise_ts : list
        Rise times, i.e. onset-peak time difference.
    half_rec : list
        Half Recovery times, i.e. time between peak and 63% amplitude.
    six_rec : list
        63 % recovery times, i.e. time between peak and 50% amplitude.

    """

    # ensure numpy
    assert len(signal) > 0, "len signal < 1"

    signal = np.array(signal)
    args, names = [], []

    # onsets, peaks, amps
    onsets, peaks, amps, _ = eda_events(signal, filt=filt, size=size, min_amplitude=min_amplitude, sampling_rate=sampling_rate)
    args += [onsets]
    names += ['onsets']
    args += [peaks]
    names += ['peaks']
    args += [amps]
    names += ['amps']

    # phasic_rate
    try:
        phasic_rate = sampling_rate * (60. / np.diff(peaks))
    except:
        phasic_rate = None
    args += [phasic_rate]
    names += ['phasic_rate']

    # rise_ts
    try:
        rise_ts = peaks - onsets
    except:
        rise_ts = None
    args += [rise_ts]
    names += ['rise_ts']

    half_rec, six_rec = rec_times(signal, onsets, peaks)
        
    args += [half_rec]
    names += ['half_rec']

    args += [six_rec]
    names += ['six_rec']

    args = np.nan_to_num(args)
    return utils.ReturnTuple(tuple(args), tuple(names))
