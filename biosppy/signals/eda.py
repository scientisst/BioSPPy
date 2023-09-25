# -*- coding: utf-8 -*-
"""
biosppy.signals.eda
-------------------

This module provides methods to process Electrodermal Activity (EDA)
signals, also known as Galvanic Skin Response (GSR).

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
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


def eda(signal=None, sampling_rate=1000., units=None, path=None, show=True):
    """Process a raw EDA signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    units : str, optional
        The units of the input signal. If specified, the plot will have the
        y-axis labeled with the corresponding units.
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
    edr : array
        Electrodermal response (EDR) signal.
    edl : array
        Electrodermal level (EDL) signal.
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
    filtered, _ = st.smoother(signal=aux,
                              kernel="boxzen",
                              size=sm_size,
                              mirror=True)

    # get SCR info
    onsets, peaks, amplitudes, phasic_rate, rise_times, half_rec, six_rec = eda_events(signal=filtered,
                                                                                       sampling_rate=sampling_rate,
                                                                                       min_amplitude=0.1, size=0.9)

    # get time vectors
    length = len(signal)
    t = (length - 1) / sampling_rate
    ts = np.linspace(0, t, length, endpoint=True)

    # get EDR and EDL
    edl_signal, edr_signal = biosppy_decomposition(signal=filtered,
                                                   sampling_rate=sampling_rate,
                                                   method="onsets",
                                                   onsets=onsets)

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
            units=units,
            path=path,
            show=True,
        )

    # output
    args = (ts, filtered, edr_signal, edl_signal, onsets, peaks, amplitudes,
            phasic_rate, rise_times, half_rec, six_rec)
    names = ("ts", "filtered", "edr", "edl", "onsets", "peaks", "amplitudes",
             "phasic_rate", "rise_times", "half_rec", "six_rec")

    return utils.ReturnTuple(args, names)


def eda_events(signal=None, sampling_rate=1000., method="emotiphai", **kwargs):
    """Returns characteristic EDA events.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
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
    rise_times : array
        Rise times, i.e. onset-peak time difference.
    half_rec : array
        Half Recovery times, i.e. time between peak and 63% amplitude.
    six_rec : array
        63 % recovery times, i.e. time between peak and 50% amplitude.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # compute onsets, peaks and amplitudes
    if method == "emotiphai":
        onsets, peaks, amps = emotiphai_eda(signal=signal,
                                            sampling_rate=sampling_rate,
                                            **kwargs)

    elif method == "kbk":
        onsets, peaks, amps = kbk_scr(signal=signal,
                                      sampling_rate=sampling_rate,
                                      **kwargs)

    elif method == "basic":
        onsets, peaks, amps = basic_scr(signal=signal)

    else:
        raise TypeError("Please specify a supported method.")

    # compute phasic rate
    try:
        phasic_rate = sampling_rate * (60. / np.diff(peaks))
    except Exception as e:
        print(e)
        phasic_rate = None

    # compute rise times
    rise_times = (peaks - onsets) / sampling_rate  # to seconds

    # compute half and 63% recovery times
    half_rec, six_rec = rec_times(signal=signal,
                                  sampling_rate=sampling_rate,
                                  onsets=onsets,
                                  peaks=peaks)

    args = (onsets, peaks, amps, phasic_rate, rise_times,
            half_rec, six_rec)
    names = ("onsets", "peaks", "amplitudes", "phasic_rate", "rise_times",
             "half_rec", "six_rec")

    return utils.ReturnTuple(args, names)


def biosppy_decomposition(signal=None, sampling_rate=1000.0, method="smoother",
                          onsets=None, **kwargs):
    """Extracts EDL and EDR signals using either a smoothing filter or onsets'
    interpolation.

    Parameters
    ----------
    signal : array
        Input filtered EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    method: str, optional
        Method to compute the edl signal: "smoother" to compute a smoothing
        filter; "onsets" to obtain edl by onsets' interpolation.
    onsets : array, optional
        List of onsets for the interpolation method.
    kwargs : dict, optional
        window_size : Size of the smoother kernel (seconds).

    Returns
    -------
    edl : array
        Electrodermal level (EDL) signal.
    edr : array
        Electrodermal response (EDR) signal.

    References
    ----------
    .. [KiBK04] K.H. Kim, S.W. Bang, and S.R. Kim, "Emotion recognition system
    using short-term monitoring of physiological signals", Med. Biol. Eng.
    Comput., vol. 42, pp. 419-427, 2004

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if method == "onsets" and onsets is None:
        raise TypeError("Please specify 'onsets' to use the onset "
                        "interpolation method.")

    # smooth method
    if method == "smoother":
        window_size = kwargs['window_size'] if 'window_size' in kwargs else 10.0
        size = int(window_size * sampling_rate)
        edl_signal, _ = st.smoother(signal=signal,
                                    kernel="bartlett",
                                    size=size,
                                    mirror=True)

    # interpolation method
    elif method == "onsets":
        # get time vectors
        length = len(signal)
        t = (length - 1) / sampling_rate
        ts = np.linspace(0, t, length, endpoint=True)

        # extract edl
        edl_on = np.hstack((ts[0], ts[onsets], ts[-1]))
        edl_amp = np.hstack((signal[0], signal[onsets], signal[-1]))
        f = interpolate.interp1d(edl_on, edl_amp)
        edl_signal = f(ts)

    else:
        raise TypeError("Please specify a supported method.")

    # differentiation
    df = np.diff(signal)

    # smooth
    size = int(1.0 * sampling_rate)
    edr_signal, _ = st.smoother(signal=df,
                                kernel="bartlett",
                                size=size,
                                mirror=True)

    # output
    args = (edl_signal, edr_signal)
    names = ("edl", "edr")

    return utils.ReturnTuple(args, names)


def cvx_decomposition(signal=None, sampling_rate=1000.0, tau0=2., tau1=0.7,
                      delta_knot=10., alpha=8e-4, gamma=1e-2, solver=None,
                      options={'reltol': 1e-9}):
    """Performs EDA decomposition using the cvxEDA algorithm.

    This function was originally developed by Luca Citi and Alberto Greco. You
    can find the original code and repository at:
    https://github.com/lciti/cvxEDA

    If you use this function in your work, please cite the original authors
    as follows: A Greco, G Valenza, A Lanata, EP Scilingo, and L Citi
    "cvxEDA: a Convex Optimization Approach to Electrodermal Activity
    Processing" IEEE Transactions on Biomedical Engineering, 2015.

    This function is used under the terms of the GNU General Public License
    v3.0 (GPLv3). You should comply with the GPLv3 if you use this code (see
    'License' section below).

    Copyright (C) 2014-2015 Luca Citi, Alberto Greco

    Parameters
    ----------
    signal : array
        Observed EDA signal (we recommend normalizing it: y = zscore(y))
    sampling_rate : int, float
        Sampling frequency (Hz).
    tau0 : float
        Slow time constant of the Bateman function
    tau1 : float
        Fast time constant of the Bateman function
    delta_knot: float
        Time between knots of the tonic spline function
    alpha: float
        Penalization for the sparse SMNA driver
    gamma : float
        Penalization for the tonic spline coefficients
    solver : ndarray
        Sparse QP solver to be used, see cvxopt.solvers.qp
    options : dict 
        solver options, see: http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
    
    Returns
    -------
    edr : array
        Phasic component
    smna : array
        Sparse SMNA driver of phasic component
    edl : array
        Tonic component
    tonic_coeff : array
        Coefficients of tonic spline
    linear_drift : array
        Offset and slope of the linear drift term
    res : array
        Model residuals
    obj : array 
        Value of objective function being minimized (eq 15 of paper)
    
    References
    ----------
    .. [cvxEDA] A Greco, G Valenza, A Lanata, EP Scilingo, and L Citi
    "cvxEDA: a Convex Optimization Approach to Electrodermal Activity
    Processing" IEEE Transactions on Biomedical Engineering, 2015. DOI:
    10.1109/TBME.2015.2474131
    
    .. [Figner2011] Figner, Bernd & Murphy, Ryan. (2011). Using skin
    conductance in judgment and decision making research. A Handbook of
    Process Tracing Methods for Decision Research.

    License
    -------
    The cvxEDA function is distributed under the GNU General Public License
    v3.0 (GPLv3). For details, please see the full license text at:
    https://www.gnu.org/licenses/gpl-3.0.en.html

    This code is provided as-is, without any warranty or support from the
    original authors.

    Notes
    -----
    Changes from original code:
    - 'y' -> 'signal'
    - 'delta' -> 1. / 'sampling_rate'
    """
    # try to import cvxopt
    try:
        import cvxopt as cv
    except ImportError:
        raise ImportError("The 'cvxopt' module is required for this function "
                          "to run. Please install it first (`pip install "
                          "cvxopt`).")

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    n = len(signal)
    y = cv.matrix(signal)
    delta = 1. / sampling_rate  # sampling interval in seconds

    # bateman ARMA model
    a1 = 1. / min(tau1, tau0)  # a1 > a0
    a0 = 1. / max(tau1, tau0)
    ar = np.array([(a1 * delta + 2.) * (a0 * delta + 2.), 2. * a1 * a0 * delta ** 2 - 8.,
                   (a1 * delta - 2.) * (a0 * delta - 2.)]) / ((a1 - a0) * delta ** 2)
    ma = np.array([1., 2., 1.])

    # matrices for ARMA model
    i = np.arange(2, n)
    A = cv.spmatrix(np.tile(ar, (n - 2, 1)), np.c_[i, i, i], np.c_[i, i - 1, i - 2], (n, n))
    M = cv.spmatrix(np.tile(ma, (n - 2, 1)), np.c_[i, i, i], np.c_[i, i - 1, i - 2], (n, n))

    # spline
    delta_knot_s = int(round(delta_knot / delta))
    spl = np.r_[np.arange(1., delta_knot_s), np.arange(delta_knot_s, 0., -1.)]  # order 1
    spl = np.convolve(spl, spl, 'full')
    spl /= max(spl)
    # matrix of spline regressors
    i = np.c_[np.arange(-(len(spl) // 2), (len(spl) + 1) // 2)] + np.r_[np.arange(0, n, delta_knot_s)]
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl), 1))
    p = np.tile(spl, (nB, 1)).T
    valid = (i >= 0) & (i < n)
    B = cv.spmatrix(p[valid], i[valid], j[valid])

    # trend
    C = cv.matrix(np.c_[np.ones(n), np.arange(1., n + 1.) / n])
    nC = C.size[1]

    # Solve the problem:
    # .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*p + .5*gamma*l'*l
    # s.t. A*q >= 0

    old_options = cv.solvers.options.copy()
    cv.solvers.options.clear()
    cv.solvers.options.update(options)
    if solver == 'conelp':
        # Use conelp
        z = lambda m, n: cv.spmatrix([], [], [], (m, n))
        G = cv.sparse([[-A, z(2, n), M, z(nB + 2, n)], [z(n + 2, nC), C, z(nB + 2, nC)],
                       [z(n, 1), -1, 1, z(n + nB + 2, 1)], [z(2 * n + 2, 1), -1, 1, z(nB, 1)],
                       [z(n + 2, nB), B, z(2, nB), cv.spmatrix(1.0, range(nB), range(nB))]])
        h = cv.matrix([z(n, 1), .5, .5, y, .5, .5, z(nB, 1)])
        c = cv.matrix([(cv.matrix(alpha, (1, n)) * A).T, z(nC, 1), 1, gamma, z(nB, 1)])
        res = cv.solvers.conelp(c, G, h, dims={'l': n, 'q': [n + 2, nB + 2], 's': []})
        obj = res['primal objective']
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cv.sparse([[Mt * M, Ct * M, Bt * M], [Mt * C, Ct * C, Bt * C],
                       [Mt * B, Ct * B, Bt * B + gamma * cv.spmatrix(1.0, range(nB), range(nB))]])
        f = cv.matrix([(cv.matrix(alpha, (1, n)) * A).T - Mt * y, -(Ct * y), -(Bt * y)])
        res = cv.solvers.qp(H, f, cv.spmatrix(-A.V, A.I, A.J, (n, len(f))),
                            cv.matrix(0., (n, 1)), solver=solver)
        obj = res['primal objective'] + .5 * (y.T * y)
    cv.solvers.options.clear()
    cv.solvers.options.update(old_options)

    l = res['x'][-nB:]
    d = res['x'][n:n + nC]
    t = B * l + C * d
    q = res['x'][:n]
    p = A * q
    r = M * q
    e = y - r - t

    # output
    args = list(np.array(a).ravel() for a in (r, p, t, l, d, e, obj))
    names = ("edr", "smna", "edl", "tonic_coeff", "linear_drift", "res", "obj")
    
    return utils.ReturnTuple(args, names)


def basic_scr(signal=None):
    """Basic method to extract Skin Conductivity Responses (SCR) from an
    EDA signal.

    Follows the approach in [Gamb08]_.

    Parameters
    ----------
    signal : array
        Input filtered EDA signal.

    Returns
    -------
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SCR peaks.
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
        Indices of the SCR peaks.
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
    df = biosppy_decomposition(signal, sampling_rate=sampling_rate)['edr']

    # zero crosses
    (zeros,) = st.zero_cross(signal=df, detrend=False)
    if np.all(df[: zeros[0]] > 0):
        zeros = zeros[1:]
    if np.all(df[zeros[-1]:] > 0):
        zeros = zeros[:-1]

    scrs, amps, ZC, peaks = [], [], [], []
    for i in range(0, len(zeros) - 1, 2):
        scrs += [df[zeros[i]: zeros[i + 1]]]
        ZC += [zeros[i]]
        ZC += [zeros[i + 1]]
        peaks += [zeros[i] + np.argmax(df[zeros[i]: zeros[i + 1]])]
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


def emotiphai_eda(signal=None, sampling_rate=1000., min_amplitude=0.1,
                  filt=True, size=1.):
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
        Indices of the SCR peaks.
    amplitudes : array
        SCR pulse amplitudes.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

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
            if (s[p] - s[0]) > min_amplitude:  # only count events with minimum amplitude
                peaks += [zeros[z] + p]
                onsets += [zeros[z]]
                amps += [s[p] - s[0]]

    # convert to array
    onsets, peaks, amps = np.array(onsets), np.array(peaks), np.array(amps)

    # output
    args = (onsets, peaks, amps)
    names = ("onsets", "peaks", "amplitudes")

    return utils.ReturnTuple(args, names)


def rec_times(signal=None, sampling_rate=1000., onsets=None, peaks=None):
    """Returns EDA recovery times.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SCR peaks.

    Returns
    -------
    half_rec : list
        Half Recovery times, i.e. time between peak and 50% amplitude.
    six_rec : list
        63 % recovery times, i.e. time between peak and 63% amplitude.

    """

    # ensure input format
    peaks = np.array(peaks, dtype=int)
    onsets = np.array(onsets, dtype=int)

    amps = np.array(signal[peaks[:]] - signal[onsets[:]])  # SCR amplitudes
    li = min(len(onsets), len(peaks))

    half_rec, six_rec = [], []
    for i in range(li):  # iterate over onset
        half_rec_amp = 0.5 * amps[i] + signal[onsets][i]
        six_rec_amp = 0.37 * amps[i] + signal[onsets][i]
        try:
            wind = np.array(signal[peaks[i]:onsets[i + 1]])
        except:
            wind = np.array(signal[peaks[i]:])  # last peak to end of signal
        half_rec_idx = np.argwhere(wind <= half_rec_amp)
        six_rec_idx = np.argwhere(wind <= six_rec_amp)
        
        if len(half_rec_idx) > 0:
            half_rec += [half_rec_idx[0][0] / sampling_rate]
        else:
            half_rec += [None]

        if len(six_rec_idx) > 0:
            six_rec += [six_rec_idx[0][0] / sampling_rate]
        else:
            six_rec += [None]

    # convert to numpy
    half_rec = np.array(half_rec)
    six_rec = np.array(six_rec)

    # output
    names = ("half_rec", "six_rec")
    args = (half_rec, six_rec)

    return utils.ReturnTuple(args, names)
