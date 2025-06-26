# -*- coding: utf-8 -*-
"""
biosppy.signals.ecg
-------------------

This module provides methods to process Electrocardiographic (ECG) signals.
Implemented code assumes a single-channel Lead I like ECG signal.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range, zip
import inspect

# 3rd party
import math
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
from scipy import stats, integrate, interpolate
import peakutils
from scipy.ndimage import filters


# local
from . import tools as st
from .. import plotting, utils
from scipy.signal import argrelextrema


def call_segmenter(segmenter, filtered_signal, sampling_rate, verbose=False, **kwargs):
    """Checks **kwargs against valid optional parameters for the selected segmenter
    and calls it with the valid arguments.

    Parameters
    ----------
    segmenter : Callable
        The segmentation function to call.
    filtered_signal : array_like
        Filtered ECG signal.
    sampling_rate : int or float
        Sampling frequency in Hz.
    verbose : bool, optional
        If True, prints which valid parameters were passed to the segmenter.
    **kwargs : dict
        Additional optional keyword arguments to pass to the segmenter.

    Returns
    -------
    rpeaks : array
        Indices of detected R-peaks.
    """
    sig = inspect.signature(segmenter)
    allowed_args = sig.parameters.keys()
    valid_args = {kwarg: val for kwarg, val in kwargs.items() if kwarg in allowed_args}
    if verbose:
        print(f"Passed valid parameters for segmenter: {segmenter}: {valid_args}")
    (rpeaks,) = segmenter(
        signal=filtered_signal, sampling_rate=sampling_rate, **valid_args
    )
    return (rpeaks,)


def ecg(
    signal=None,
    sampling_rate=1000.0,
    units=None,
    path=None,
    show=True,
    interactive=False,
    segmenter="hamilton",
    verbose_segmenting=False,
    **kwargs,
):
    """Process a raw ECG signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw ECG signal.
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
    segmenter : string, optional
        Name of the segmenting algorithm.
        Available options:

        - `"hamilton"` → `hamilton_segmenter`
        - `"ASI"` → `ASI_segmenter`
        - `"gamboa"` → `gamboa_segmenter`
        - `"engzee"` → `engzee_segmenter`
        - `"ssf"` → `ssf_segmenter`
        - `"pan-tompkins"` → `Pan_Tompkins_Plus_Plus_segmenter`
        If a different name is passed hamilton will be used.
    verbose_segmenting : bool, optional
        If True, prints which valid parameters (for the chosen segmenter)
        were passed.
    **kwargs : dict
        Additional optional keyword arguments (currently only viable for
        segmenter parameters).


    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered ECG signal.
    rpeaks : array
        R-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).

    """

    ALLOWED_KWARGS = ["Pth", "threshold", "tol", "before", "after"]
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # validate kwargs
    invalid_args = [k for k in kwargs if k not in ALLOWED_KWARGS]
    if invalid_args:
        raise ValueError(f"Unknown argument in kwargs: {invalid_args}")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    order = int(1.5 * sampling_rate)
    filtered, _, _ = st.filter_signal(
        signal=signal,
        ftype="FIR",
        band="bandpass",
        order=order,
        frequency=[0.67, 45],
        sampling_rate=sampling_rate,
    )

    filtered = filtered - np.mean(filtered)  # remove DC offset

    # segment
    segmenters = {
        "hamilton": hamilton_segmenter,
        "ASI": ASI_segmenter,
        "gamboa": gamboa_segmenter,
        "engzee": engzee_segmenter,
        "ssf": ssf_segmenter,
        "pan-tompkins": Pan_Tompkins_Plus_Plus_segmenter,
    }
    try:
        chosen_segmenter = segmenters[segmenter]
        (rpeaks,) = call_segmenter(
            chosen_segmenter, filtered, sampling_rate, verbose_segmenting, **kwargs
        )
    except KeyError:
        print("Unrecognized segmenter chosen, will use default (hamilton)")
        (rpeaks,) = hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)

    # correct R-peak locations
    (rpeaks,) = correct_rpeaks(
        signal=filtered, rpeaks=rpeaks, sampling_rate=sampling_rate, tol=0.05
    )

    # extract templates
    templates, rpeaks = extract_heartbeats(
        signal=filtered,
        rpeaks=rpeaks,
        sampling_rate=sampling_rate,
        before=0.2,
        after=0.4,
    )

    # compute heart rate
    hr_idx, hr = st.get_heart_rate(
        beats=rpeaks, sampling_rate=sampling_rate, smooth=True, size=3
    )

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_hr = ts[hr_idx]
    ts_tmpl = np.linspace(-0.2, 0.4, templates.shape[1], endpoint=False)

    # plot
    if show:
        if interactive:
            from biosppy.inter_plotting import ecg as inter_plotting

            inter_plotting.plot_ecg(
                ts=ts,
                raw=signal,
                filtered=filtered,
                rpeaks=rpeaks,
                templates_ts=ts_tmpl,
                templates=templates,
                heart_rate_ts=ts_hr,
                heart_rate=hr,
                path=path,
                show=True,
            )

        else:
            plotting.plot_ecg(
                ts=ts,
                raw=signal,
                filtered=filtered,
                rpeaks=rpeaks,
                templates_ts=ts_tmpl,
                templates=templates,
                heart_rate_ts=ts_hr,
                heart_rate=hr,
                units=units,
                path=path,
                show=True,
            )

    # output
    args = (ts, filtered, rpeaks, ts_tmpl, templates, ts_hr, hr)
    names = (
        "ts",
        "filtered",
        "rpeaks",
        "templates_ts",
        "templates",
        "heart_rate_ts",
        "heart_rate",
    )

    return utils.ReturnTuple(args, names)


def _extract_heartbeats(signal=None, rpeaks=None, before=200, after=400):
    """Extract heartbeat templates from an ECG signal, given a list of
    R-peak locations.

    Parameters
    ----------
    signal : array
        Input ECG signal.
    rpeaks : array
        R-peak location indices.
    before : int, optional
        Number of samples to include before the R peak.
    after : int, optional
        Number of samples to include after the R peak.

    Returns
    -------
    templates : array
        Extracted heartbeat templates.
    rpeaks : array
        Corresponding R-peak location indices of the extracted heartbeat
        templates.

    """

    R = np.sort(rpeaks)
    length = len(signal)
    templates = []
    newR = []

    for r in R:
        a = r - before
        if a < 0:
            continue
        b = r + after
        if b > length:
            break
        templates.append(signal[a:b])
        newR.append(r)

    templates = np.array(templates)
    newR = np.array(newR, dtype="int")

    return templates, newR


def extract_heartbeats(
    signal=None, rpeaks=None, sampling_rate=1000.0, before=0.2, after=0.4
):
    """Extract heartbeat templates from an ECG signal, given a list of
    R-peak locations.

    Parameters
    ----------
    signal : array
        Input ECG signal.
    rpeaks : array
        R-peak location indices.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    before : float, optional
        Window size to include before the R peak (seconds).
    after : int, optional
        Window size to include after the R peak (seconds).

    Returns
    -------
    templates : array
        Extracted heartbeat templates.
    rpeaks : array
        Corresponding R-peak location indices of the extracted heartbeat
        templates.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if rpeaks is None:
        raise TypeError("Please specify the input R-peak locations.")

    if before < 0:
        raise ValueError("Please specify a non-negative 'before' value.")
    if after < 0:
        raise ValueError("Please specify a non-negative 'after' value.")

    # convert delimiters to samples
    before = int(before * sampling_rate)
    after = int(after * sampling_rate)

    # get heartbeats
    templates, newR = _extract_heartbeats(
        signal=signal, rpeaks=rpeaks, before=before, after=after
    )

    return utils.ReturnTuple((templates, newR), ("templates", "rpeaks"))


def compare_segmentation(
    reference=None, test=None, sampling_rate=1000.0, offset=0, minRR=None, tol=0.05
):
    """Compare the segmentation performance of a list of R-peak positions
    against a reference list.

    Parameters
    ----------
    reference : array
        Reference R-peak location indices.
    test : array
        Test R-peak location indices.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    offset : int, optional
        Constant a priori offset (number of samples) between reference and
        test R-peak locations.
    minRR : float, optional
        Minimum admissible RR interval (seconds).
    tol : float, optional
        Tolerance between corresponding reference and test R-peak
        locations (seconds).

    Returns
    -------
    TP : int
        Number of true positive R-peaks.
    FP : int
        Number of false positive R-peaks.
    performance : float
        Test performance; TP / len(reference).
    acc : float
        Accuracy rate; TP / (TP + FP).
    err : float
        Error rate; FP / (TP + FP).
    match : list
        Indices of the elements of 'test' that match to an R-peak
        from 'reference'.
    deviation : array
        Absolute errors of the matched R-peaks (seconds).
    mean_deviation : float
        Mean error (seconds).
    std_deviation : float
        Standard deviation of error (seconds).
    mean_ref_ibi : float
        Mean of the reference interbeat intervals (seconds).
    std_ref_ibi : float
        Standard deviation of the reference interbeat intervals (seconds).
    mean_test_ibi : float
        Mean of the test interbeat intervals (seconds).
    std_test_ibi : float
        Standard deviation of the test interbeat intervals (seconds).

    """

    # check inputs
    if reference is None:
        raise TypeError(
            "Please specify an input reference list of R-peak \
                        locations."
        )

    if test is None:
        raise TypeError(
            "Please specify an input test list of R-peak \
                        locations."
        )

    if minRR is None:
        minRR = np.inf

    sampling_rate = float(sampling_rate)

    # ensure numpy
    reference = np.array(reference)
    test = np.array(test)

    # convert to samples
    minRR = minRR * sampling_rate
    tol = tol * sampling_rate

    TP = 0
    FP = 0

    matchIdx = []
    dev = []

    for i, r in enumerate(test):
        # deviation to closest R in reference
        ref = reference[np.argmin(np.abs(reference - (r + offset)))]
        error = np.abs(ref - (r + offset))

        if error < tol:
            TP += 1
            matchIdx.append(i)
            dev.append(error)
        else:
            if len(matchIdx) > 0:
                bdf = r - test[matchIdx[-1]]
                if bdf < minRR:
                    # false positive, but removable with RR interval check
                    pass
                else:
                    FP += 1
            else:
                FP += 1

    # convert deviations to time
    dev = np.array(dev, dtype="float")
    dev /= sampling_rate
    nd = len(dev)
    if nd == 0:
        mdev = np.nan
        sdev = np.nan
    elif nd == 1:
        mdev = np.mean(dev)
        sdev = 0.0
    else:
        mdev = np.mean(dev)
        sdev = np.std(dev, ddof=1)

    # interbeat interval
    th1 = 1.5  # 40 bpm
    th2 = 0.3  # 200 bpm

    rIBI = np.diff(reference)
    rIBI = np.array(rIBI, dtype="float")
    rIBI /= sampling_rate

    good = np.nonzero((rIBI < th1) & (rIBI > th2))[0]
    rIBI = rIBI[good]

    nr = len(rIBI)
    if nr == 0:
        rIBIm = np.nan
        rIBIs = np.nan
    elif nr == 1:
        rIBIm = np.mean(rIBI)
        rIBIs = 0.0
    else:
        rIBIm = np.mean(rIBI)
        rIBIs = np.std(rIBI, ddof=1)

    tIBI = np.diff(test[matchIdx])
    tIBI = np.array(tIBI, dtype="float")
    tIBI /= sampling_rate

    good = np.nonzero((tIBI < th1) & (tIBI > th2))[0]
    tIBI = tIBI[good]

    nt = len(tIBI)
    if nt == 0:
        tIBIm = np.nan
        tIBIs = np.nan
    elif nt == 1:
        tIBIm = np.mean(tIBI)
        tIBIs = 0.0
    else:
        tIBIm = np.mean(tIBI)
        tIBIs = np.std(tIBI, ddof=1)

    # output
    perf = float(TP) / len(reference)
    acc = float(TP) / (TP + FP)
    err = float(FP) / (TP + FP)

    args = (
        TP,
        FP,
        perf,
        acc,
        err,
        matchIdx,
        dev,
        mdev,
        sdev,
        rIBIm,
        rIBIs,
        tIBIm,
        tIBIs,
    )
    names = (
        "TP",
        "FP",
        "performance",
        "acc",
        "err",
        "match",
        "deviation",
        "mean_deviation",
        "std_deviation",
        "mean_ref_ibi",
        "std_ref_ibi",
        "mean_test_ibi",
        "std_test_ibi",
    )

    return utils.ReturnTuple(args, names)


def correct_rpeaks(signal=None, rpeaks=None, sampling_rate=1000.0, tol=0.05):
    """Correct R-peak locations to the maximum within a tolerance.

    Parameters
    ----------
    signal : array
        ECG signal.
    rpeaks : array
        R-peak location indices.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    tol : int, float, optional
        Correction tolerance (seconds).

    Returns
    -------
    rpeaks : array
        Cerrected R-peak location indices.

    Notes
    -----
    * The tolerance is defined as the time interval :math:`[R-tol, R+tol[`.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if rpeaks is None:
        raise TypeError("Please specify the input R-peaks.")

    tol = int(tol * sampling_rate)
    length = len(signal)

    newR = []
    for r in rpeaks:
        a = r - tol
        if a < 0:
            continue
        b = r + tol
        if b > length:
            break
        newR.append(a + np.argmax(signal[a:b]))

    newR = sorted(list(set(newR)))
    newR = np.array(newR, dtype="int")

    return utils.ReturnTuple((newR,), ("rpeaks",))


def ssf_segmenter(
    signal=None, sampling_rate=1000.0, threshold=20, before=0.03, after=0.01
):
    """ECG R-peak segmentation based on the Slope Sum Function (SSF).

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    threshold : float, optional
        SSF threshold.
    before : float, optional
        Search window size before R-peak candidate (seconds).
    after : float, optional
        Search window size after R-peak candidate (seconds).

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # convert to samples
    winB = int(before * sampling_rate)
    winA = int(after * sampling_rate)

    Rset = set()
    length = len(signal)

    # diff
    dx = np.diff(signal)
    dx[dx >= 0] = 0
    dx = dx**2

    # detection
    (idx,) = np.nonzero(dx > threshold)
    idx0 = np.hstack(([0], idx))
    didx = np.diff(idx0)

    # search
    sidx = idx[didx > 1]
    for item in sidx:
        a = item - winB
        if a < 0:
            a = 0
        b = item + winA
        if b > length:
            continue

        r = np.argmax(signal[a:b]) + a
        Rset.add(r)

    # output
    rpeaks = list(Rset)
    rpeaks.sort()
    rpeaks = np.array(rpeaks, dtype="int")

    return utils.ReturnTuple((rpeaks,), ("rpeaks",))


def christov_segmenter(signal=None, sampling_rate=1000.0):
    """ECG R-peak segmentation algorithm.

    Follows the approach by Christov [Chri04]_.

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    References
    ----------
    .. [Chri04] Ivaylo I. Christov, "Real time electrocardiogram QRS
       detection using combined adaptive threshold", BioMedical Engineering
       OnLine 2004, vol. 3:28, 2004

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    length = len(signal)

    # algorithm parameters
    v100ms = int(0.1 * sampling_rate)
    v50ms = int(0.050 * sampling_rate)
    v300ms = int(0.300 * sampling_rate)
    v350ms = int(0.350 * sampling_rate)
    v200ms = int(0.2 * sampling_rate)
    v1200ms = int(1.2 * sampling_rate)
    M_th = 0.4  # paper is 0.6

    # Pre-processing
    # 1. Moving averaging filter for power-line interference suppression:
    # averages samples in one period of the powerline
    # interference frequency with a first zero at this frequency.
    b = np.ones(int(0.02 * sampling_rate)) / 50.0
    a = [1]
    X = ss.filtfilt(b, a, signal)
    # 2. Moving averaging of samples in 28 ms interval for electromyogram
    # noise suppression a filter with first zero at about 35 Hz.
    b = np.ones(int(sampling_rate / 35.0)) / 35.0
    X = ss.filtfilt(b, a, X)
    X, _, _ = st.filter_signal(
        signal=X,
        ftype="butter",
        band="lowpass",
        order=7,
        frequency=40.0,
        sampling_rate=sampling_rate,
    )
    X, _, _ = st.filter_signal(
        signal=X,
        ftype="butter",
        band="highpass",
        order=7,
        frequency=9.0,
        sampling_rate=sampling_rate,
    )

    k, Y, L = 1, [], len(X)
    for n in range(k + 1, L - k):
        Y.append(X[n] ** 2 - X[n - k] * X[n + k])
    Y = np.array(Y)
    Y[Y < 0] = 0

    # Complex lead
    # Y = abs(scipy.diff(X)) # 1-lead
    # 3. Moving averaging of a complex lead (the sintesis is
    # explained in the next section) in 40 ms intervals a filter
    # with first zero at about 25 Hz. It is suppressing the noise
    # magnified by the differentiation procedure used in the
    # process of the complex lead sintesis.
    b = np.ones(int(sampling_rate / 25.0)) / 25.0
    Y = ss.lfilter(b, a, Y)

    # Init
    MM = M_th * np.max(Y[: int(5 * sampling_rate)]) * np.ones(5)
    MMidx = 0
    M = np.mean(MM)
    slope = np.linspace(1.0, 0.6, int(sampling_rate))
    Rdec = 0
    R = 0
    RR = np.zeros(5)
    RRidx = 0
    Rm = 0
    QRS = []
    Rpeak = []
    current_sample = 0
    skip = False
    F = np.mean(Y[:v350ms])

    # Go through each sample
    while current_sample < len(Y):
        if QRS:
            # No detection is allowed 200 ms after the current one. In
            # the interval QRS to QRS+200ms a new value of M5 is calculated: newM5 = 0.6*max(Yi)
            if current_sample <= QRS[-1] + v200ms:
                Mnew = M_th * max(Y[QRS[-1] : QRS[-1] + v200ms])
                # The estimated newM5 value can become quite high, if
                # steep slope premature ventricular contraction or artifact
                # appeared, and for that reason it is limited to newM5 = 1.1*M5 if newM5 > 1.5* M5
                # The MM buffer is refreshed excluding the oldest component, and including M5 = newM5.
                Mnew = Mnew if Mnew <= 1.5 * MM[MMidx - 1] else 1.1 * MM[MMidx - 1]
                MM[MMidx] = Mnew
                MMidx = np.mod(MMidx + 1, 5)
                # M is calculated as an average value of MM.
                Mtemp = np.mean(MM)
                M = Mtemp
                skip = True
            # M is decreased in an interval 200 to 1200 ms following
            # the last QRS detection at a low slope, reaching 60 % of its
            # refreshed value at 1200 ms.
            elif (
                current_sample >= QRS[-1] + v200ms
                and current_sample < QRS[-1] + v1200ms
            ):
                M = Mtemp * slope[current_sample - QRS[-1] - v200ms]
            # After 1200 ms M remains unchanged.
            # R = 0 V in the interval from the last detected QRS to 2/3 of the expected Rm.
            if current_sample >= QRS[-1] and current_sample < QRS[-1] + (2 / 3.0) * Rm:
                R = 0
            # In the interval QRS + Rm * 2/3 to QRS + Rm, R decreases
            # 1.4 times slower then the decrease of the previously discussed
            # steep slope threshold (M in the 200 to 1200 ms interval).
            elif (
                current_sample >= QRS[-1] + (2 / 3.0) * Rm
                and current_sample < QRS[-1] + Rm
            ):
                R += Rdec
            # After QRS + Rm the decrease of R is stopped
            # MFR = M + F + R
        MFR = M + F + R
        # QRS or beat complex is detected if Yi = MFR
        if not skip and Y[current_sample] >= MFR:
            QRS += [current_sample]
            Rpeak += [QRS[-1] + np.argmax(Y[QRS[-1] : QRS[-1] + v300ms])]
            if len(QRS) >= 2:
                # A buffer with the 5 last RR intervals is updated at any new QRS detection.
                RR[RRidx] = QRS[-1] - QRS[-2]
                RRidx = np.mod(RRidx + 1, 5)
        skip = False
        # With every signal sample, F is updated adding the maximum
        # of Y in the latest 50 ms of the 350 ms interval and
        # subtracting maxY in the earliest 50 ms of the interval.
        if current_sample >= v350ms:
            Y_latest50 = Y[current_sample - v50ms : current_sample]
            Y_earliest50 = Y[current_sample - v350ms : current_sample - v300ms]
            F += (max(Y_latest50) - max(Y_earliest50)) / 1000.0
        # Rm is the mean value of the buffer RR.
        Rm = np.mean(RR)
        current_sample += 1

    rpeaks = []
    for i in Rpeak:
        a, b = i - v100ms, i + v100ms
        if a < 0:
            a = 0
        if b > length:
            b = length
        rpeaks.append(np.argmax(signal[a:b]) + a)

    rpeaks = sorted(list(set(rpeaks)))
    rpeaks = np.array(rpeaks, dtype="int")

    return utils.ReturnTuple((rpeaks,), ("rpeaks",))


def engzee_segmenter(signal=None, sampling_rate=1000.0, threshold=0.48):
    """ECG R-peak segmentation algorithm.

    Follows the approach by Engelse and Zeelenberg [EnZe79]_ with the
    modifications by Lourenco *et al.* [LSLL12]_.

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    threshold : float, optional
        Detection threshold.

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    References
    ----------
    .. [EnZe79] W. Engelse and C. Zeelenberg, "A single scan algorithm for
       QRS detection and feature extraction", IEEE Comp. in Cardiology,
       vol. 6, pp. 37-42, 1979
    .. [LSLL12] A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred,
       "Real Time Electrocardiogram Segmentation for Finger Based ECG
       Biometrics", BIOSIGNALS 2012, pp. 49-54, 2012

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # algorithm parameters
    changeM = int(0.75 * sampling_rate)
    Miterate = int(1.75 * sampling_rate)
    v250ms = int(0.25 * sampling_rate)
    v1200ms = int(1.2 * sampling_rate)
    v1500ms = int(1.5 * sampling_rate)
    v180ms = int(0.18 * sampling_rate)
    p10ms = int(np.ceil(0.01 * sampling_rate))
    p20ms = int(np.ceil(0.02 * sampling_rate))
    err_kill = int(0.01 * sampling_rate)
    inc = 1
    mmth = threshold
    mmp = 0.2

    # Differentiator (1)
    y1 = [signal[i] - signal[i - 4] for i in range(4, len(signal))]

    # Low pass filter (2)
    c = [1, 4, 6, 4, 1, -1, -4, -6, -4, -1]
    y2 = np.array([np.dot(c, y1[n - 9 : n + 1]) for n in range(9, len(y1))])
    y2_len = len(y2)

    # vars
    MM = mmth * max(y2[:Miterate]) * np.ones(3)
    MMidx = 0
    Th = np.mean(MM)
    NN = mmp * min(y2[:Miterate]) * np.ones(2)
    NNidx = 0
    ThNew = np.mean(NN)
    update = False
    nthfpluss = []
    rpeaks = []

    # Find nthf+ point
    while True:
        # If a previous intersection was found, continue the analysis from there
        if update:
            if inc * changeM + Miterate < y2_len:
                a = (inc - 1) * changeM
                b = inc * changeM + Miterate
                Mnew = mmth * max(y2[a:b])
                Nnew = mmp * min(y2[a:b])
            elif y2_len - (inc - 1) * changeM > v1500ms:
                a = (inc - 1) * changeM
                Mnew = mmth * max(y2[a:])
                Nnew = mmp * min(y2[a:])
            if len(y2) - inc * changeM > Miterate:
                MM[MMidx] = Mnew if Mnew <= 1.5 * MM[MMidx - 1] else 1.1 * MM[MMidx - 1]
                NN[NNidx] = (
                    Nnew
                    if abs(Nnew) <= 1.5 * abs(NN[NNidx - 1])
                    else 1.1 * NN[NNidx - 1]
                )
            MMidx = np.mod(MMidx + 1, len(MM))
            NNidx = np.mod(NNidx + 1, len(NN))
            Th = np.mean(MM)
            ThNew = np.mean(NN)
            inc += 1
            update = False
        if nthfpluss:
            lastp = nthfpluss[-1] + 1
            if lastp < (inc - 1) * changeM:
                lastp = (inc - 1) * changeM
            y22 = y2[lastp : inc * changeM + err_kill]
            # find intersection with Th
            try:
                nthfplus = np.intersect1d(
                    np.nonzero(y22 > Th)[0], np.nonzero(y22 < Th)[0] - 1
                )[0]
            except IndexError:
                if inc * changeM > len(y2):
                    break
                else:
                    update = True
                    continue
            # adjust index
            nthfplus += int(lastp)
            # if a previous R peak was found:
            if rpeaks:
                # check if intersection is within the 200-1200 ms interval. Modification: 300 ms -> 200 bpm
                if nthfplus - rpeaks[-1] > v250ms and nthfplus - rpeaks[-1] < v1200ms:
                    pass
                # if new intersection is within the <200ms interval, skip it. Modification: 300 ms -> 200 bpm
                elif nthfplus - rpeaks[-1] < v250ms:
                    nthfpluss += [nthfplus]
                    continue
        # no previous intersection, find the first one
        else:
            try:
                aux = np.nonzero(
                    y2[(inc - 1) * changeM : inc * changeM + err_kill] > Th
                )[0]
                bux = (
                    np.nonzero(y2[(inc - 1) * changeM : inc * changeM + err_kill] < Th)[
                        0
                    ]
                    - 1
                )
                nthfplus = int((inc - 1) * changeM) + np.intersect1d(aux, bux)[0]
            except IndexError:
                if inc * changeM > len(y2):
                    break
                else:
                    update = True
                    continue
        nthfpluss += [nthfplus]
        # Define 160ms search region
        windowW = np.arange(nthfplus, nthfplus + v180ms)
        # Check if the condition y2[n] < Th holds for a specified
        # number of consecutive points (experimentally we found this number to be at least 10 points)"
        i, f = windowW[0], windowW[-1] if windowW[-1] < len(y2) else -1
        hold_points = np.diff(np.nonzero(y2[i:f] < ThNew)[0])
        cont = 0
        for hp in hold_points:
            if hp == 1:
                cont += 1
                if cont == p10ms - 1:  # -1 is because diff eats a sample
                    max_shift = p20ms  # looks for X's max a bit to the right
                    if nthfpluss[-1] > max_shift:
                        rpeaks += [np.argmax(signal[i - max_shift : f]) + i - max_shift]
                    else:
                        rpeaks += [np.argmax(signal[i:f]) + i]
                    break
            else:
                cont = 0

    rpeaks = sorted(list(set(rpeaks)))
    rpeaks = np.array(rpeaks, dtype="int")

    return utils.ReturnTuple((rpeaks,), ("rpeaks",))


def gamboa_segmenter(signal=None, sampling_rate=1000.0, tol=0.002):
    """ECG R-peak segmentation algorithm.

    Follows the approach by Gamboa.

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    tol : float, optional
        Tolerance parameter.

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # convert to samples
    v_100ms = int(0.1 * sampling_rate)
    v_300ms = int(0.3 * sampling_rate)
    hist, edges = np.histogram(signal, 100, density=True)

    TH = 0.01
    F = np.cumsum(hist)

    v0 = edges[np.nonzero(F > TH)[0][0]]
    v1 = edges[np.nonzero(F < (1 - TH))[0][-1]]

    nrm = max([abs(v0), abs(v1)])
    norm_signal = signal / float(nrm)

    d2 = np.diff(norm_signal, 2)

    b = np.nonzero((np.diff(np.sign(np.diff(-d2)))) == -2)[0] + 2
    b = np.intersect1d(b, np.nonzero(-d2 > tol)[0])

    if len(b) < 3:
        rpeaks = []
    else:
        b = b.astype("float")
        rpeaks = []
        previous = b[0]
        for i in b[1:]:
            if i - previous > v_300ms:
                previous = i
                rpeaks.append(np.argmax(signal[int(i) : int(i + v_100ms)]) + i)

    rpeaks = sorted(list(set(rpeaks)))
    rpeaks = np.array(rpeaks, dtype="int")

    return utils.ReturnTuple((rpeaks,), ("rpeaks",))


def hamilton_segmenter(signal=None, sampling_rate=1000.0):
    """ECG R-peak segmentation algorithm.

    Follows the approach by Hamilton [Hami02]_.

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    References
    ----------
    .. [Hami02] P.S. Hamilton, "Open Source ECG Analysis Software
       Documentation", E.P.Limited, 2002

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    sampling_rate = float(sampling_rate)
    length = len(signal)
    dur = length / sampling_rate

    # algorithm parameters
    v1s = int(1.0 * sampling_rate)
    v100ms = int(0.1 * sampling_rate)
    TH_elapsed = np.ceil(0.36 * sampling_rate)
    sm_size = int(0.08 * sampling_rate)
    init_ecg = 8  # seconds for initialization
    if dur < init_ecg:
        init_ecg = int(dur)

    # filtering
    filtered, _, _ = st.filter_signal(
        signal=signal,
        ftype="butter",
        band="lowpass",
        order=4,
        frequency=25.0,
        sampling_rate=sampling_rate,
    )
    filtered, _, _ = st.filter_signal(
        signal=filtered,
        ftype="butter",
        band="highpass",
        order=4,
        frequency=3.0,
        sampling_rate=sampling_rate,
    )

    # diff
    dx = np.abs(np.diff(filtered, 1) * sampling_rate)

    # smoothing
    dx, _ = st.smoother(signal=dx, kernel="hamming", size=sm_size, mirror=True)

    # buffers
    qrspeakbuffer = np.zeros(init_ecg)
    noisepeakbuffer = np.zeros(init_ecg)
    peak_idx_test = np.zeros(init_ecg)
    noise_idx = np.zeros(init_ecg)
    rrinterval = sampling_rate * np.ones(init_ecg)

    a, b = 0, v1s
    all_peaks, _ = st.find_extrema(signal=dx, mode="max")
    for i in range(init_ecg):
        peaks, values = st.find_extrema(signal=dx[a:b], mode="max")
        try:
            ind = np.argmax(values)
        except ValueError:
            pass
        else:
            # peak amplitude
            qrspeakbuffer[i] = values[ind]
            # peak location
            peak_idx_test[i] = peaks[ind] + a

        a += v1s
        b += v1s

    # thresholds
    ANP = np.median(noisepeakbuffer)
    AQRSP = np.median(qrspeakbuffer)
    TH = 0.475
    DT = ANP + TH * (AQRSP - ANP)
    DT_vec = []
    indexqrs = 0
    indexnoise = 0
    indexrr = 0
    npeaks = 0
    offset = 0

    beats = []

    # detection rules
    # 1 - ignore all peaks that precede or follow larger peaks by less than 200ms
    lim = int(np.ceil(0.2 * sampling_rate))
    diff_nr = int(np.ceil(0.045 * sampling_rate))
    bpsi, bpe = offset, 0

    for f in all_peaks:
        DT_vec += [DT]
        # 1 - Checking if f-peak is larger than any peak following or preceding it by less than 200 ms
        peak_cond = np.array(
            (all_peaks > f - lim) * (all_peaks < f + lim) * (all_peaks != f)
        )
        peaks_within = all_peaks[peak_cond]
        if peaks_within.any() and (max(dx[peaks_within]) > dx[f]):
            continue

        # 4 - If the peak is larger than the detection threshold call it a QRS complex, otherwise call it noise
        if dx[f] > DT:
            # 2 - look for both positive and negative slopes in raw signal
            if f < diff_nr:
                diff_now = np.diff(signal[0 : f + diff_nr])
            elif f + diff_nr >= len(signal):
                diff_now = np.diff(signal[f - diff_nr : len(dx)])
            else:
                diff_now = np.diff(signal[f - diff_nr : f + diff_nr])
            diff_signer = diff_now[diff_now > 0]
            if len(diff_signer) == 0 or len(diff_signer) == len(diff_now):
                continue
            # RR INTERVALS
            if npeaks > 0:
                # 3 - in here we check point 3 of the Hamilton paper
                # that is, we check whether our current peak is a valid R-peak.
                prev_rpeak = beats[npeaks - 1]

                elapsed = f - prev_rpeak
                # if the previous peak was within 360 ms interval
                if elapsed < TH_elapsed:
                    # check current and previous slopes
                    if prev_rpeak < diff_nr:
                        diff_prev = np.diff(signal[0 : prev_rpeak + diff_nr])
                    elif prev_rpeak + diff_nr >= len(signal):
                        diff_prev = np.diff(signal[prev_rpeak - diff_nr : len(dx)])
                    else:
                        diff_prev = np.diff(
                            signal[prev_rpeak - diff_nr : prev_rpeak + diff_nr]
                        )

                    slope_now = max(diff_now)
                    slope_prev = max(diff_prev)

                    if slope_now < 0.5 * slope_prev:
                        # if current slope is smaller than half the previous one, then it is a T-wave
                        continue
                if dx[f] < 3.0 * np.median(qrspeakbuffer):  # avoid retarded noise peaks
                    beats += [int(f) + bpsi]
                else:
                    continue

                if bpe == 0:
                    rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                    indexrr += 1
                    if indexrr == init_ecg:
                        indexrr = 0
                else:
                    if beats[npeaks] > beats[bpe - 1] + v100ms:
                        rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                        indexrr += 1
                        if indexrr == init_ecg:
                            indexrr = 0

            elif dx[f] < 3.0 * np.median(qrspeakbuffer):
                beats += [int(f) + bpsi]
            else:
                continue

            npeaks += 1
            qrspeakbuffer[indexqrs] = dx[f]
            peak_idx_test[indexqrs] = f
            indexqrs += 1
            if indexqrs == init_ecg:
                indexqrs = 0
        if dx[f] <= DT:
            # 4 - not valid
            # 5 - If no QRS has been detected within 1.5 R-to-R intervals,
            # there was a peak that was larger than half the detection threshold,
            # and the peak followed the preceding detection by at least 360 ms,
            # classify that peak as a QRS complex
            tf = f + bpsi
            # RR interval median
            RRM = np.median(rrinterval)  # initial values are good?

            if len(beats) >= 2:
                elapsed = tf - beats[npeaks - 1]

                if elapsed >= 1.5 * RRM and elapsed > TH_elapsed:
                    if dx[f] > 0.5 * DT:
                        beats += [int(f) + offset]
                        # RR INTERVALS
                        if npeaks > 0:
                            rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                            indexrr += 1
                            if indexrr == init_ecg:
                                indexrr = 0
                        npeaks += 1
                        qrspeakbuffer[indexqrs] = dx[f]
                        peak_idx_test[indexqrs] = f
                        indexqrs += 1
                        if indexqrs == init_ecg:
                            indexqrs = 0
                else:
                    noisepeakbuffer[indexnoise] = dx[f]
                    noise_idx[indexnoise] = f
                    indexnoise += 1
                    if indexnoise == init_ecg:
                        indexnoise = 0
            else:
                noisepeakbuffer[indexnoise] = dx[f]
                noise_idx[indexnoise] = f
                indexnoise += 1
                if indexnoise == init_ecg:
                    indexnoise = 0

        # Update Detection Threshold
        ANP = np.median(noisepeakbuffer)
        AQRSP = np.median(qrspeakbuffer)
        DT = ANP + 0.475 * (AQRSP - ANP)

    beats = np.array(beats)

    r_beats = []
    thres_ch = 0.85
    adjacency = 0.05 * sampling_rate
    for i in beats:
        error = [False, False]
        if i - lim < 0:
            window = signal[0 : i + lim]
            add = 0
        elif i + lim >= length:
            window = signal[i - lim : length]
            add = i - lim
        else:
            window = signal[i - lim : i + lim]
            add = i - lim
        # meanval = np.mean(window)
        w_peaks, _ = st.find_extrema(signal=window, mode="max")
        w_negpeaks, _ = st.find_extrema(signal=window, mode="min")
        zerdiffs = np.where(np.diff(window) == 0)[0]
        w_peaks = np.concatenate((w_peaks, zerdiffs))
        w_negpeaks = np.concatenate((w_negpeaks, zerdiffs))

        pospeaks = sorted(zip(window[w_peaks], w_peaks), reverse=True)
        negpeaks = sorted(zip(window[w_negpeaks], w_negpeaks))

        try:
            twopeaks = [pospeaks[0]]
        except IndexError:
            twopeaks = []
        try:
            twonegpeaks = [negpeaks[0]]
        except IndexError:
            twonegpeaks = []

        # getting positive peaks
        for i in range(len(pospeaks) - 1):
            if abs(pospeaks[0][1] - pospeaks[i + 1][1]) > adjacency:
                twopeaks.append(pospeaks[i + 1])
                break
        try:
            posdiv = abs(twopeaks[0][0] - twopeaks[1][0])
        except IndexError:
            error[0] = True

        # getting negative peaks
        for i in range(len(negpeaks) - 1):
            if abs(negpeaks[0][1] - negpeaks[i + 1][1]) > adjacency:
                twonegpeaks.append(negpeaks[i + 1])
                break
        try:
            negdiv = abs(twonegpeaks[0][0] - twonegpeaks[1][0])
        except IndexError:
            error[1] = True

        # choosing type of R-peak
        n_errors = sum(error)
        try:
            if not n_errors:
                if posdiv > thres_ch * negdiv:
                    # pos noerr
                    r_beats.append(twopeaks[0][1] + add)
                else:
                    # neg noerr
                    r_beats.append(twonegpeaks[0][1] + add)
            elif n_errors == 2:
                if abs(twopeaks[0][1]) > abs(twonegpeaks[0][1]):
                    # pos allerr
                    r_beats.append(twopeaks[0][1] + add)
                else:
                    # neg allerr
                    r_beats.append(twonegpeaks[0][1] + add)
            elif error[0]:
                # pos poserr
                r_beats.append(twopeaks[0][1] + add)
            else:
                # neg negerr
                r_beats.append(twonegpeaks[0][1] + add)
        except IndexError:
            continue

    rpeaks = sorted(list(set(r_beats)))
    rpeaks = np.array(rpeaks, dtype="int")

    return utils.ReturnTuple((rpeaks,), ("rpeaks",))


def ASI_segmenter(signal=None, sampling_rate=1000.0, Pth=5.0):
    """ECG R-peak segmentation algorithm.

    Parameters
    ----------
    signal : array
        Input ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    Pth : int, float, optional
        Free parameter used in exponential decay

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    References
    ----------
    Modification by Tiago Rodrigues, based on:
    [R. Gutiérrez-rivas 2015] Novel Real-Time Low-Complexity QRS Complex Detector
                            Based on Adaptive Thresholding. Vol. 15,no. 10, pp. 6036–6043, 2015.
    [D. Sadhukhan]  R-Peak Detection Algorithm for Ecg using Double Difference
                    And RRInterval Processing. Procedia Technology, vol. 4, pp. 873–877, 2012.

    """

    N = round(3 * sampling_rate / 128)
    Nd = N - 1
    Rmin = 0.26

    rpeaks = []
    i = 1
    tf = len(signal)
    Ramptotal = 0

    # Double derivative squared
    diff_ecg = [signal[i] - signal[i - Nd] for i in range(Nd, len(signal))]
    ddiff_ecg = [diff_ecg[i] - diff_ecg[i - 1] for i in range(1, len(diff_ecg))]
    squar = np.square(ddiff_ecg)

    # Integrate moving window
    b = np.array(np.ones(N))
    a = [1]
    processed_ecg = ss.lfilter(b, a, squar)

    # R-peak finder FSM
    while i < tf - sampling_rate:  # ignore last second of recording

        # State 1: looking for maximum
        tf1 = round(i + Rmin * sampling_rate)
        Rpeakamp = 0
        while i < tf1:
            # Rpeak amplitude and position
            if processed_ecg[i] > Rpeakamp:
                Rpeakamp = processed_ecg[i]
                rpeakpos = i + 1
            i += 1

        Ramptotal = (19 / 20) * Ramptotal + (1 / 20) * Rpeakamp
        rpeaks.append(rpeakpos)

        # State 2: waiting state
        d = tf1 - rpeakpos
        tf2 = i + round(0.2 * 250 - d)
        while i <= tf2:
            i += 1

        # State 3: decreasing threshold
        Thr = Ramptotal
        while processed_ecg[i] < Thr:
            Thr = Thr * math.exp(-Pth / sampling_rate)
            i += 1

    return utils.ReturnTuple((rpeaks,), ("rpeaks",))


def Pan_Tompkins_Plus_Plus_segmenter(signal=None, sampling_rate=1000.0):
    """ECG QRS-Peak Detection Algorithm

    Follows the approach by Md Niaz and Naimul [MdNiNai22]_.

    Parameters
    ----------

    signal : array
        Input raw ECG signal.
    sampling_rate: int, float, optional
        Sampling frequency (Hz).

    Returns
    ----------
    qrs_i_raw: array
        R-peak location indices.

    References
    ----------
    .. [MdNiNai22] Khan, Naimul and Imtiaz, Md Niaz, "Pan-Tompkins++: A Robust Approach to Detect R-peaks in ECG Signals",
        arXiv preprint arXiv:2211.03171, 2022

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    """ Initialize """
    delay = 0
    skip = 0  # Becomes one when a T wave is detected
    m_selected_RR = 0
    mean_RR = 0
    ser_back = 0

    """ Noise Cancelation (Filtering) (5-18 Hz) """

    if sampling_rate == 200:
        """Remove the mean of Signal"""
        # If fs=200 keep frequency 5-12 Hz otherwise 5-18 Hz
        signal = signal - np.mean(signal)

        Wn = 12 * 2 / sampling_rate
        N = 3
        a, b = ss.butter(N, Wn, btype="lowpass")
        ecg_l = ss.filtfilt(a, b, signal)

        ecg_l = ecg_l / np.max(
            np.abs(ecg_l)
        )  # Normalize by dividing high value. This reduces time of calculation

        Wn = 5 * 2 / sampling_rate
        N = 3  # Order of 3 less processing
        a, b = signal.butter(N, Wn, btype="highpass")  # Bandpass filtering
        ecg_h = signal.filtfilt(a, b, ecg_l, padlen=3 * (max(len(a), len(b)) - 1))
        ecg_h = ecg_h / np.max(
            np.abs(ecg_h)
        )  # Normalize by dividing high value. This reduces time of calculation

    else:
        """Band Pass Filter for noise cancelation of other sampling frequencies (Filtering)"""
        f1 = 5  # 3 #5                                          # cutoff low frequency to get rid of baseline wander
        f2 = 18  # 25  #15                                         # cutoff frequency to discard high frequency noise
        Wn = [f1 * 2 / sampling_rate, f2 * 2 / sampling_rate]  # cutoff based on fs
        N = 3
        a, b = ss.butter(N=N, Wn=Wn, btype="bandpass")  # Bandpass filtering
        ecg_h = ss.filtfilt(a, b, signal, padlen=3 * (max(len(a), len(b)) - 1))

        ecg_h = ecg_h / np.max(np.abs(ecg_h))

    vector = [1, 2, 0, -2, -1]
    if sampling_rate != 200:
        int_c = 160 / sampling_rate
        b = interpolate.interp1d(range(1, 6), [i * sampling_rate / 8 for i in vector])(
            np.arange(1, 5.1, int_c)
        )

    else:
        b = [i * sampling_rate / 8 for i in vector]

    ecg_d = ss.filtfilt(b, 1, ecg_h, padlen=3 * (max(len(a), len(b)) - 1))

    ecg_d = ecg_d / np.max(ecg_d)

    """ Squaring nonlinearly enhance the dominant peaks """

    ecg_s = ecg_d**2

    # Smoothing
    sm_size = int(0.06 * sampling_rate)
    ecg_s = st.smoother(signal=ecg_s, kernel="flattop", size=sm_size, mirror=True)

    temp_vector = np.ones((1, round(0.150 * sampling_rate))) / round(
        0.150 * sampling_rate
    )  # 150ms moving window, widest possible QRS width
    temp_vector = temp_vector.flatten()
    ecg_m = np.convolve(
        ecg_s["signal"], temp_vector
    )  # Convolution signal and moving window sample

    delay = delay + round(0.150 * sampling_rate) / 2

    pks = []
    locs = peakutils.indexes(
        y=ecg_m, thres=0, min_dist=round(0.231 * sampling_rate)
    )  # Find all the peaks apart from previous peak 231ms, peak indices
    for val in locs:
        pks.append(ecg_m[val])  # Peak magnitudes

    """ Initialize Some Other Parameters """
    LLp = len(pks)

    """ Stores QRS with respect to Signal and Filtered Signal """
    qrs_c = np.zeros(
        LLp
    )  # Amplitude of R peak in convoluted (after moving window) signal
    qrs_i = np.zeros(LLp)  # Index of R peak in convoluted (after moving window) signal
    qrs_i_raw = np.zeros(
        LLp
    )  # Index of R peak in filtered (before derivative and moving windoe) signal
    qrs_amp_raw = np.zeros(LLp)  # Amplitude of R in filtered signal
    """ Noise Buffers """
    nois_c = np.zeros(LLp)
    nois_i = np.zeros(LLp)

    """ Buffers for signal and noise """

    SIGL_buf = np.zeros(LLp)
    NOISL_buf = np.zeros(LLp)
    THRS_buf = np.zeros(LLp)
    SIGL_buf1 = np.zeros(LLp)
    NOISL_buf1 = np.zeros(LLp)
    THRS_buf1 = np.zeros(LLp)

    """ Initialize the training phase (2 seconds of the signal) to determine the THR_SIG and THR_NOISE """
    # Threshold of signal after moving average operation; Take first 2s window max peak to set initial Threshold
    THR_SIG = (
        np.max(ecg_m[: 2 * int(sampling_rate) + 1]) * 1 / 3
    )  # Threshold-1 (paper) #0.33 of the max amplitude
    THR_NOISE = (
        np.mean(ecg_m[: 2 * int(sampling_rate) + 1]) * 1 / 2
    )  # Threshold-2 (paper) # 0.5 of the mean signal is considered to be noise
    SIG_LEV = THR_SIG  # SPK for convoluted (after moving window) signal
    NOISE_LEV = THR_NOISE  # NPK for convoluted (after moving window) signal

    """ Initialize bandpath filter threshold (2 seconds of the bandpass signal) """
    # Threshold of signal before derivative and moving average operation, just after 5-18 Hz filtering
    THR_SIG1 = np.max(ecg_h[: 2 * int(sampling_rate) + 1]) * 1 / 3  # Threshold-1
    THR_NOISE1 = np.mean(ecg_h[: 2 * int(sampling_rate) + 1]) * 1 / 2  # Threshold-2
    SIG_LEV1 = THR_SIG1  # Signal level in Bandpassed filter; SPK for filtered signal
    NOISE_LEV1 = THR_NOISE1  # Noise level in Bandpassed filter; NPK for filtered signal

    """ Thresholding and decision rule """

    Beat_C = 0  # Beat count for convoluted signal
    Beat_C1 = 0  # Beat count for filtred signal
    Noise_Count = 0
    Check_Flag = 0
    for i in range(LLp):
        """Locate the corresponding peak in the filtered signal"""

        if locs[i] - round(0.150 * sampling_rate) >= 1 and locs[i] <= len(ecg_h):
            temp_vec = ecg_h[
                locs[i] - round(0.150 * sampling_rate) : locs[i] + 1
            ]  # Find the values from the preceding 150ms of the peak
            y_i = np.max(temp_vec)  # Find the max magnitude in that 150ms window
            x_i = list(temp_vec).index(
                y_i
            )  # Find the index of the max value with respect to (peak-150ms) starts as 0 index
        else:
            if i == 0:
                temp_vec = ecg_h[: locs[i] + 1]
                y_i = np.max(temp_vec)
                x_i = list(temp_vec).index(y_i)
                ser_back = 1
            elif locs[i] >= len(ecg_h):
                temp_vec = ecg_h[int(locs[i] - round(0.150 * sampling_rate)) :]  # c
                y_i = np.max(temp_vec)
                x_i = list(temp_vec).index(y_i)

        """ Update the Hearth Rate """
        if Beat_C >= 9:
            diffRR = np.diff(
                qrs_i[Beat_C - 9 : Beat_C]
            )  # Calculate RR interval of recent 8 heart beats (taken from R peaks)
            mean_RR = np.mean(
                diffRR
            )  # Calculate the mean of 8 previous R waves interval
            comp = qrs_i[Beat_C - 1] - qrs_i[Beat_C - 2]  # Latest RR

            m_selected_RR = mean_RR  # The latest regular beats mean

        """ Calculate the mean last 8 R waves """
        if bool(m_selected_RR):
            test_m = m_selected_RR  # if the regular RR available use it
        elif bool(mean_RR) and m_selected_RR == 0:
            test_m = mean_RR
        else:
            test_m = 0

        # If no R peaks in 1.4s then check with the reduced Threshold
        if (locs[i] - qrs_i[Beat_C - 1]) >= round(1.4 * sampling_rate):

            temp_vec = ecg_m[
                int(qrs_i[Beat_C - 1] + round(0.360 * sampling_rate)) : int(locs[i]) + 1
            ]  # Search after 360ms of previous QRS to current peak
            if temp_vec.size:
                pks_temp = np.max(
                    temp_vec
                )  # search back and locate the max in the interval
                locs_temp = list(temp_vec).index(pks_temp)
                locs_temp = qrs_i[Beat_C - 1] + round(0.360 * sampling_rate) + locs_temp

                if pks_temp > THR_NOISE * 0.2:  # Check with 20% of the noise threshold

                    Beat_C = Beat_C + 1
                    if (Beat_C - 1) >= LLp:
                        break
                    qrs_c[Beat_C - 1] = pks_temp
                    qrs_i[Beat_C - 1] = locs_temp

                    """ Locate in Filtered Signal """
                    # Once we find the peak in convoluted signal, we will search in the filtered signal for max peak with a 150 ms window before that location
                    if locs_temp <= len(ecg_h):

                        temp_vec = ecg_h[
                            int(locs_temp - round(0.150 * sampling_rate))
                            + 1 : int(locs_temp)
                            + 2
                        ]
                        y_i_t = np.max(temp_vec)
                        x_i_t = list(temp_vec).index(y_i_t)
                    else:
                        temp_vec = ecg_h[
                            int(locs_temp - round(0.150 * sampling_rate)) :
                        ]
                        y_i_t = np.max(temp_vec)
                        x_i_t = list(temp_vec).index(y_i_t)

                    if y_i_t > THR_NOISE1 * 0.2:
                        Beat_C1 = Beat_C1 + 1
                        if (Beat_C1 - 1) >= LLp:
                            break
                        temp_value = locs_temp - round(0.150 * sampling_rate) + x_i_t
                        qrs_i_raw[Beat_C1 - 1] = temp_value
                        qrs_amp_raw[Beat_C1 - 1] = y_i_t

                        SIG_LEV1 = 0.75 * y_i_t + 0.25 * SIG_LEV1

                    not_nois = 1

                    SIG_LEV = 0.75 * pks_temp + 0.25 * SIG_LEV
            else:
                not_nois = 0

        elif bool(
            test_m
        ):  # Check for missed QRS if no QRS is detected in 166 percent of
            # the current average RR interval or 1s after the last detected QRS. the maximal peak detected in
            # that time interval that lies Threshold1 and Threshold-3 (paper) is considered to be a possible QRS complex

            if ((locs[i] - qrs_i[Beat_C - 1]) >= round(1.66 * test_m)) or (
                (locs[i] - qrs_i[Beat_C - 1]) > round(1 * sampling_rate)
            ):  # it shows a QRS is missed

                temp_vec = ecg_m[
                    int(qrs_i[Beat_C - 1] + round(0.360 * sampling_rate)) : int(locs[i])
                    + 1
                ]  # Search after 360ms of previous QRS to current peak
                if temp_vec.size:
                    pks_temp = np.max(
                        temp_vec
                    )  # search back and locate the max in the interval
                    locs_temp = list(temp_vec).index(pks_temp)
                    locs_temp = (
                        qrs_i[Beat_C - 1] + round(0.360 * sampling_rate) + locs_temp
                    )

                    # Consider signal between the preceding 3 QRS complexes and the following 3 peaks to calculate Threshold-3 (paper)

                    THR_NOISE_TMP = THR_NOISE
                    if i < (len(locs) - 3):
                        temp_vec_tmp = ecg_m[
                            int(qrs_i[Beat_C - 3] + round(0.360 * sampling_rate)) : int(
                                locs[i + 3]
                            )
                            + 1
                        ]  # values between the preceding 3 QRS complexes and the following 3 peaks
                        THR_NOISE_TMP = 0.5 * THR_NOISE + 0.5 * (
                            np.mean(temp_vec_tmp) * 1 / 2
                        )  # Calculate Threshold3

                    if (
                        pks_temp > THR_NOISE_TMP
                    ):  # If max peak in that range greater than Threshold3 mark that as a heart beat

                        Beat_C = Beat_C + 1
                        if (Beat_C - 1) >= LLp:
                            break
                        qrs_c[Beat_C - 1] = (
                            pks_temp  # Mark R peak in the convoluted signal
                        )
                        qrs_i[Beat_C - 1] = locs_temp

                        """ Locate in Filtered Signal """
                        # Once we find the peak in convoluted signal, we will search in the filtered signal for max peak with a 150 ms window before that location
                        if locs_temp <= len(ecg_h):

                            temp_vec = ecg_h[
                                int(locs_temp - round(0.150 * sampling_rate))
                                + 1 : int(locs_temp)
                                + 2
                            ]
                            y_i_t = np.max(temp_vec)
                            x_i_t = list(temp_vec).index(y_i_t)
                        else:
                            temp_vec = ecg_h[
                                int(locs_temp - round(0.150 * sampling_rate)) :
                            ]
                            y_i_t = np.max(temp_vec)
                            x_i_t = list(temp_vec).index(y_i_t)

                        """ Band Pass Signal Threshold """
                        THR_NOISE1_TMP = THR_NOISE1
                        if i < (len(locs) - 3):
                            temp_vec_tmp = ecg_h[
                                int(
                                    qrs_i[Beat_C - 3]
                                    + round(0.360 * sampling_rate)
                                    - round(0.150 * sampling_rate)
                                    + 1
                                ) : int(locs[i + 3])
                                + 1
                            ]
                            THR_NOISE1_TMP = 0.5 * THR_NOISE1 + 0.5 * (
                                np.mean(temp_vec_tmp) * 1 / 2
                            )
                        if y_i_t > THR_NOISE1_TMP:
                            Beat_C1 = Beat_C1 + 1
                            if (Beat_C1 - 1) >= LLp:
                                break
                            temp_value = (
                                locs_temp - round(0.150 * sampling_rate) + x_i_t
                            )
                            qrs_i_raw[Beat_C1 - 1] = (
                                temp_value  # R peak marked with index in filtered signal
                            )
                            qrs_amp_raw[Beat_C1 - 1] = y_i_t  # Amplitude of that R peak

                            SIG_LEV1 = 0.75 * y_i_t + 0.25 * SIG_LEV1

                        not_nois = 1
                        # Changed- For missed R peaks- Update THR
                        SIG_LEV = 0.75 * pks_temp + 0.25 * SIG_LEV
                else:
                    not_nois = 0
            else:
                not_nois = 0

        """ Find noise and QRS Peaks """

        if pks[i] >= THR_SIG:
            """if NO QRS in 360 ms of the previous QRS or in 50 percent of
            the current average RR interval, See if T wave"""

            if Beat_C >= 3:
                if bool(test_m):
                    if (locs[i] - qrs_i[Beat_C - 1]) <= round(
                        0.5 * test_m
                    ):  # Check 50 percent of the current average RR interval

                        Check_Flag = 1
                if (
                    locs[i] - qrs_i[Beat_C - 1] <= round(0.36 * sampling_rate)
                ) or Check_Flag == 1:

                    temp_vec = ecg_m[
                        locs[i] - round(0.07 * sampling_rate) : locs[i] + 1
                    ]
                    Slope1 = np.mean(
                        np.diff(temp_vec)
                    )  # mean slope of the waveform at that position
                    temp_vec = ecg_m[
                        int(qrs_i[Beat_C - 1] - round(0.07 * sampling_rate))
                        - 1 : int(qrs_i[Beat_C - 1])
                        + 1
                    ]
                    Slope2 = np.mean(np.diff(temp_vec))  # mean slope of previous R wave

                    if np.abs(Slope1) <= np.abs(
                        0.6 * Slope2
                    ):  # slope less then 0.6 of previous R; checking if it is noise
                        Noise_Count = Noise_Count + 1
                        nois_c[Noise_Count] = pks[i]
                        nois_i[Noise_Count] = locs[i]
                        skip = 1  # T wave identification
                    else:
                        skip = 0

            """ Skip is 1 when a T wave is detected """
            if skip == 0:
                Beat_C = Beat_C + 1
                if (Beat_C - 1) >= LLp:
                    break
                qrs_c[Beat_C - 1] = pks[i]  # Mark as R peak in the convoluted signal
                qrs_i[Beat_C - 1] = locs[i]

                """ Band pass Filter check threshold """

                if y_i >= THR_SIG1:
                    Beat_C1 = Beat_C1 + 1  # Mark as R peak in the filtered signal
                    if (Beat_C1 - 1) >= LLp:
                        break
                    if bool(ser_back):
                        # +1 to agree with Matlab implementation
                        temp_value = x_i + 1
                        qrs_i_raw[Beat_C1 - 1] = temp_value
                    else:
                        temp_value = locs[i] - round(0.150 * sampling_rate) + x_i
                        qrs_i_raw[Beat_C1 - 1] = temp_value

                    qrs_amp_raw[Beat_C1 - 1] = y_i

                    SIG_LEV1 = 0.125 * y_i + 0.875 * SIG_LEV1

                SIG_LEV = 0.125 * pks[i] + 0.875 * SIG_LEV

        elif THR_NOISE <= pks[i] and pks[i] < THR_SIG:
            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
            NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV

        elif (
            pks[i] < THR_NOISE
        ):  # If less than noise threshold (Threshold-2) mark as noise
            nois_c[Noise_Count] = pks[i]
            nois_i[Noise_Count] = locs[i]
            Noise_Count = Noise_Count + 1

            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
            NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV

        """ Adjust the threshold with SNR """

        if NOISE_LEV != 0 or SIG_LEV != 0:
            THR_SIG = NOISE_LEV + 0.25 * (
                np.abs(SIG_LEV - NOISE_LEV)
            )  # Calculate Threshold-1 for convoluted signal; above this R peak
            THR_NOISE = (
                0.4 * THR_SIG
            )  # Calculate Threshold-2 for convoluted signal; below this Noise

        """ Adjust the threshold with SNR for bandpassed signal """

        if NOISE_LEV1 != 0 or SIG_LEV1 != 0:
            THR_SIG1 = NOISE_LEV1 + 0.25 * (
                np.abs(SIG_LEV1 - NOISE_LEV1)
            )  # Calculate Threshold-1  for filtered signal; above this R peak
            THR_NOISE1 = (
                0.4 * THR_SIG1
            )  # Calculate Threshold-2 for filtered signal; below this Noise

        """ take a track of thresholds of smoothed signal """

        SIGL_buf[i] = SIG_LEV
        NOISL_buf[i] = NOISE_LEV
        THRS_buf[i] = THR_SIG

        """ take a track of thresholds of filtered signal """

        SIGL_buf1[i] = SIG_LEV1
        NOISL_buf1[i] = NOISE_LEV1
        THRS_buf1[i] = THR_SIG1

        """ reset parameters """

        skip = 0
        not_nois = 0
        ser_back = 0
        Check_Flag = 0

    """ Adjust lengths """

    qrs_i_raw = qrs_i_raw[:Beat_C1]
    qrs_amp_raw = qrs_amp_raw[:Beat_C1]
    qrs_c = qrs_c[: Beat_C + 1]
    qrs_i = qrs_i[: Beat_C + 1]

    return utils.ReturnTuple((qrs_i_raw.astype(int),), ("rpeaks",))


def find_artifacts(peaks, sampling_rate):
    """find_artifacts: find and classify artifacts

    Parameters
    ----------
        peaks: array
            Vector containing indices of detected peaks (R waves locations)
        sampling_rate : float
            ECG sampling frequency, in Hz.

    Returns
    -------
        artifacts: dictionary
            Struct containing indices of detected artifacts.
        subspaces: dictionary
            Subspaces containing rr, drrs, mrrs, s12, s22, c1, c2 used to classify artifacts.
    """
    c1 = 0.13
    c2 = 0.17
    alpha = 5.2
    ww = 91
    medfilt_order = 11

    rr = np.diff(peaks) / sampling_rate
    rr = np.insert(rr, 0, np.mean(rr))

    # Artifact identification
    drrs = np.diff(rr)
    drrs = np.insert(drrs, 0, np.mean(drrs))
    th1 = estimate_th(drrs, alpha, ww)
    drrs = drrs / th1

    padding = 2

    drrs_pad = np.pad(drrs, padding, "reflect")

    """drrs_pad = np.pad(drrs, (padding, padding), 'symmetric')
    drrs_pad[:padding] += 1
    drrs_pad[-padding:] -= 1"""

    s12 = np.zeros(len(drrs))
    for d in range(padding, padding + len(drrs)):
        if drrs_pad[d] > 0:
            s12[d - padding] = max([drrs_pad[d - 1], drrs_pad[d + 1]])
        elif drrs_pad[d] < 0:
            s12[d - padding] = min([drrs_pad[d - 1], drrs_pad[d + 1]])

    s22 = np.zeros(len(drrs))
    for d in range(padding, padding + len(drrs)):
        if drrs_pad[d] >= 0:
            s22[d - padding] = min([drrs_pad[d + 1], drrs_pad[d + 2]])
        elif drrs_pad[d] < 0:
            s22[d - padding] = max([drrs_pad[d + 1], drrs_pad[d + 2]])

    medrr = ss.medfilt(rr, medfilt_order)
    mrrs = rr - medrr
    mrrs[mrrs < 0] *= 2
    th2 = estimate_th(mrrs, alpha, ww)
    mrrs = mrrs / th2

    # Artifacts classification
    extra_indices = []
    missed_indices = []
    ectopic_indices = []
    longshort_indices = []

    i = 0
    while i < len(rr) - 2:
        if abs(drrs[i]) <= 1:
            i += 1
            continue

        eq1 = (drrs[i] > 1) and (s12[i] < (-c1 * drrs[i] - c2))
        eq2 = (drrs[i] < -1) and (s12[i] > (-c1 * drrs[i] + c2))

        if any([eq1, eq2]):
            ectopic_indices.append(i)
            i += 1
            continue

        if not any([abs(drrs[i]) > 1, abs(mrrs[i]) > 3]):
            i += 1
            continue

        longshort_candidates = [i]

        if abs(drrs[i + 1]) < abs(drrs[i + 2]):
            longshort_candidates.append(i + 1)

        for j in longshort_candidates:
            eq3 = (drrs[j] > 1) and (s22[j] < -1)
            eq4 = abs(mrrs[j]) > 3
            eq5 = (drrs[j] < -1) and (s22[j] > 1)

            if not any([eq3, eq4, eq5]):
                i += 1
                continue

            eq6 = abs(rr[j] / 2 - medrr[j]) < th2[j]
            eq7 = abs(rr[j] + rr[j + 1] - medrr[j]) < th2[j]

            if all([eq5, eq7]):
                extra_indices.append(j)
                i += 1
                continue

            if all([eq3, eq6]):
                missed_indices.append(j)
                i += 1
                continue

            longshort_indices.append(j)
            i += 1

    artifacts = {
        "ectopic": ectopic_indices,
        "missed": missed_indices,
        "extra": extra_indices,
        "longshort": longshort_indices,
    }
    subspaces = {
        "rr": rr,
        "drrs": drrs,
        "mrrs": mrrs,
        "s12": s12,
        "s22": s22,
        "c1": c1,
        "c2": c2,
    }

    return artifacts, subspaces


def estimate_th(x, alpha, ww):
    """estimate_th: estimate threshold

    Parameters
    ----------
        x: array
            Vector containing drrs or mrrs.
        alpha : float
            Empirically obtaind constant used in threshold calculation.
        ww: int
            Window width in ms.

    Returns
    -------
        th: float
            Threshold.
    """

    x_abs = np.abs(x)

    q1 = filters.percentile_filter(x_abs, 25, size=ww, mode="reflect")
    q3 = filters.percentile_filter(x_abs, 75, size=ww, mode="reflect")

    th = alpha * ((q3 - q1) / 2)

    return th


def correct_extra(extra_indices, peaks):
    """correct_extra: correct extra beat by deleting it.

    Parameters
    ----------
        extra_indices: array
            Vector containing indices of extra beats.
        peaks : array
            Vector containing indices of detected peaks (R waves locations).

    Returns
    -------
        corrected_peaks: array
            Vector containing indices of corrected peaks.
    """
    corrected_peaks = peaks.copy()
    corrected_peaks = np.delete(corrected_peaks, extra_indices)
    return corrected_peaks


def correct_misaligned(misaligned_indices, peaks):
    """correct_misaligned: correct misaligned beat (long or short) by interpolating new values to the RR time series.

    Parameters
    ----------
        misaligned_indices: array
            Vector containing indices of misaligned beats.
        peaks : array
            Vector containing indices of detected peaks (R waves locations).

    Returns
    -------
        corrected_peaks: array
            Vector containing indices of corrected peaks.
    """
    corrected_peaks = np.array(peaks.copy())

    misaligned_indices = np.array(misaligned_indices)
    valid_indices = np.logical_and(
        misaligned_indices > 1, misaligned_indices < len(corrected_peaks) - 1
    )
    misaligned_indices = misaligned_indices[valid_indices]
    prev_peaks = corrected_peaks[misaligned_indices - 1]
    next_peaks = corrected_peaks[misaligned_indices + 1]

    half_ibi = (next_peaks - prev_peaks) / 2
    peaks_interp = prev_peaks + half_ibi

    corrected_peaks = np.delete(corrected_peaks, misaligned_indices)
    corrected_peaks = np.round(np.sort(np.concatenate((corrected_peaks, peaks_interp))))

    return corrected_peaks


def correct_missed(missed_indices, peaks):
    """correct_missed: correct missed beat by adding new R-wave occurrence time so that
    it divides the detected long RR interval into two equal halves and RR interval series
    is then recalculated.

    Parameters
    ----------
        missed_indices: array
            Vector containing indices of missed beats.
        peaks : array
            Vector containing indices of detected peaks (R waves locations).

    Returns
    -------
        corrected_peaks: array
            Vector containing indices of corrected peaks.
    """
    corrected_peaks = peaks.copy()
    missed_indices = np.array(missed_indices)
    valid_indices = np.logical_and(
        missed_indices > 1, missed_indices < len(corrected_peaks)
    )
    missed_indices = missed_indices[valid_indices]
    prev_peaks = corrected_peaks[[i - 1 for i in missed_indices]]
    next_peaks = corrected_peaks[missed_indices]
    added_peaks = [
        round(prev_peaks[i] + (next_peaks[i] - prev_peaks[i]) / 2)
        for i in range(len(valid_indices))
    ]

    corrected_peaks = np.insert(corrected_peaks, missed_indices, added_peaks)

    return corrected_peaks


def update_indices(source_indices, update_indices, update):
    """update_indices: updates the indices in update_indices based on the values in source_indices and update.

    Parameters
    ----------
        source_indices: array
            Vector containing original indices.
        update_indices : array
            Vector containing update_indices.
        update: int
            Update index

    Returns
    -------
        list(np.unique(update_indices)): array
            Vector containing unique updated indices.
    """
    if not update_indices:
        return update_indices
    for s in source_indices:
        update_indices = [u + update if u > s else u for u in update_indices]
    return list(np.unique(update_indices))


def correct_artifacts(artifacts, peaks):
    """correct_artifacts: correct artifacts according to its type.

    Parameters
    ----------
        artifacts: dictionary
            Struct containing indices of detected artifacts.
        peaks: array
            Vector containing indices of detected peaks (R waves locations)

    Returns
    -------
        peaks: array
            Vector containing indices of corrected R peaks.
    """
    extra_indices = artifacts["extra"]
    missed_indices = artifacts["missed"]
    ectopic_indices = artifacts["ectopic"]
    longshort_indices = artifacts["longshort"]

    if extra_indices:
        peaks = correct_extra(extra_indices, peaks)
        missed_indices = update_indices(extra_indices, missed_indices, -1)
        ectopic_indices = update_indices(extra_indices, ectopic_indices, -1)
        longshort_indices = update_indices(extra_indices, longshort_indices, -1)

    if missed_indices:
        peaks = correct_missed(missed_indices, peaks)
        ectopic_indices = update_indices(missed_indices, ectopic_indices, 1)
        longshort_indices = update_indices(missed_indices, longshort_indices, 1)

    if ectopic_indices:
        peaks = correct_misaligned(ectopic_indices, peaks)

    if longshort_indices:
        peaks = correct_misaligned(longshort_indices, peaks)

    return peaks


def plot_artifacts(artifacts, subspaces):
    """plot_artifacts: plot artifacts according to its type.

    Parameters
    ----------
        artifacts: dictionary
            Struct containing indices of detected artifacts.
        subspaces: dictionary
            Subspaces containing rr, drrs, mrrs, s12, s22, c1, c2 used to classify artifacts.

    Returns
    -------
        None
    """
    ectopic_indices = artifacts["ectopic"]
    missed_indices = artifacts["missed"]
    extra_indices = artifacts["extra"]
    longshort_indices = artifacts["longshort"]

    rr = subspaces["rr"]
    drrs = subspaces["drrs"]
    mrrs = subspaces["mrrs"]
    s12 = subspaces["s12"]
    s22 = subspaces["s22"]
    c1 = subspaces["c1"]
    c2 = subspaces["c2"]

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Artifact types
    axs[0, 0].plot(rr, "k")
    axs[0, 0].scatter(longshort_indices, rr[longshort_indices], 15, "c")
    axs[0, 0].scatter(ectopic_indices, rr[ectopic_indices], 15, "r")
    axs[0, 0].scatter(extra_indices, rr[extra_indices], 15, "m")
    axs[0, 0].scatter(missed_indices, rr[missed_indices], 15, "g")
    axs[0, 0].legend(
        ["", "Long/Short", "Ectopic", "False positive", "False negative"],
        loc="upper right",
    )
    axs[0, 0].set_title("Artifact types")

    # Th1
    axs[1, 0].plot(abs(drrs))
    axs[1, 0].axhline(y=1, color="r", linestyle="--")
    axs[1, 0].set_title("Consecutive-difference criterion")

    # Th2
    axs[1, 1].plot(abs(mrrs))
    axs[1, 1].axhline(y=3, color="r", linestyle="--")
    axs[1, 1].set_title("Difference-from-median criterion")

    # Subspace S12
    axs[2, 0].scatter(drrs, s12, 15, "k")
    axs[2, 0].scatter(drrs[longshort_indices], s12[longshort_indices], 15, "c")
    axs[2, 0].scatter(drrs[ectopic_indices], s12[ectopic_indices], 15, "r")
    axs[2, 0].scatter(drrs[extra_indices], s12[extra_indices], 15, "m")
    axs[2, 0].scatter(drrs[missed_indices], s12[missed_indices], 15, "g")
    axs[2, 0].add_patch(
        patches.Polygon(
            [[-10, 5], [-10, -c1 * -10 + c2], [-1, -c1 * -1 + c2], [-1, 5]],
            alpha=0.05,
            color="k",
        )
    )
    axs[2, 0].add_patch(
        patches.Polygon(
            [[1, -c1 * 1 - c2], [1, -5], [10, -5], [10, -c1 * 10 - c2]],
            alpha=0.05,
            color="k",
        )
    )
    axs[2, 0].set_title("Subspace S12")

    # Subspace S21
    axs[2, 1].scatter(drrs, s22, 15, "k")
    axs[2, 1].scatter(drrs[longshort_indices], s22[longshort_indices], 15, "c")
    axs[2, 1].scatter(drrs[ectopic_indices], s22[ectopic_indices], 15, "r")
    axs[2, 1].scatter(drrs[extra_indices], s22[extra_indices], 15, "m")
    axs[2, 1].scatter(drrs[missed_indices], s22[missed_indices], 15, "g")
    axs[2, 1].add_patch(
        patches.Polygon([[-10, 10], [-10, 1], [-1, 1], [-1, 10]], alpha=0.05, color="k")
    )
    axs[2, 1].add_patch(
        patches.Polygon([[1, -1], [1, -10], [10, -10], [10, -1]], alpha=0.05, color="k")
    )
    axs[2, 1].set_title("Subspace S21")

    plt.tight_layout()
    plt.show()


# função principal
def fixpeaks(peaks, sampling_rate=1000, iterative=True, show=False):
    """FIXPEAKS: HRV time series artifact correction.

    Follows the approach by Lipponen et. al, 2019 [Lipp19].
    Matlab implementation by Marek Sokol, 2022.

    Parameters
    ----------
        peaks: array
            Vector containing indices of detected peaks (R waves locations)
        sampling_rate : int, float, optional
            ECG sampling frequency, in Hz.
        iterative: boolean, optional
            Repeatedly apply the artifact correction (default = true).
        show: boolean, optional
            Visualize artifacts and artifact thresholds (default = false).

    Returns
    -------
        artifacts: dictionary
            Struct containing indices of detected artifacts.
        peaks_clean: array
            Vector of corrected peak values (indices)

    References
    ----------
    .. [Lipp19] Jukka A. Lipponen & Mika P. Tarvainen (2019): A robust algorithm for heart rate variability
       time series artefact correction using novel beat classification, Journal of Medical Engineering & Technology
    """

    # check inputs
    if peaks is None:
        raise TypeError("Please specify an input R peaks array.")

    artifacts, subspaces = find_artifacts(peaks, sampling_rate)
    peaks_clean = correct_artifacts(artifacts, peaks)

    if iterative:
        n_artifacts_current = sum([len(v) for v in artifacts.values()])

        while True:
            new_artifacts, new_subspaces = find_artifacts(peaks_clean, sampling_rate)
            n_artifacts_previous = n_artifacts_current
            n_artifacts_current = sum([len(v) for v in new_artifacts.values()])

            if n_artifacts_current >= n_artifacts_previous:
                break

            artifacts = new_artifacts
            subspaces = new_subspaces
            peaks_clean = correct_artifacts(artifacts, peaks_clean)

    if show:
        plot_artifacts(artifacts, subspaces)

    return utils.ReturnTuple((artifacts, peaks_clean), ("artifacts", "peaks_clean"))


def getQPositions(ecg_proc=None, show=False):
    """Different ECG Waves (Q, R, S, ...) are not present or are not so clear to identify in all ECG signals (I II III V1 V2 V3, ...)
    For Q wave we suggest to use signals I, aVL . Avoid II, III, V1, V2, V3, V4, aVR, aVF

    Parameters
    ----------
    signal : object
    object return by the function ecg.
    show : bool, optional
    If True, show a plot of the Q Positions on every signal sample/template.

    Returns
    -------
    Q_positions : array
            Array with all Q positions on the signal
    Q_start_ positions : array
            Array with all Q start positions on the signal
    """

    templates_ts = ecg_proc["templates_ts"]
    template_r_position = np.argmin(
        np.abs(templates_ts - 0)
    )  # R peak on the template is always on time instant 0 seconds
    Q_positions = []
    Q_start_positions = []
    Q_positions_template = []
    Q_start_positions_template = []

    for n, each in enumerate(ecg_proc["templates"]):
        # Get Q Position
        template_left = each[0 : template_r_position + 1]
        mininums_from_template_left = argrelextrema(template_left, np.less)
        try:
            Q_position = ecg_proc["rpeaks"][n] - (
                template_r_position - mininums_from_template_left[0][-1]
            )
            Q_positions.append(Q_position)
            Q_positions_template.append(mininums_from_template_left[0][-1])
        except:
            pass

        # Get Q start position
        template_Q_left = each[0 : mininums_from_template_left[0][-1] + 1]
        maximum_from_template_Q_left = argrelextrema(template_Q_left, np.greater)
        try:
            Q_start_position = (
                ecg_proc["rpeaks"][n]
                - template_r_position
                + maximum_from_template_Q_left[0][-1]
            )
            Q_start_positions.append(Q_start_position)
            Q_start_positions_template.append(maximum_from_template_Q_left[0][-1])
        except:
            pass
    if show:
        plt.figure()
        plt.plot(ecg_proc["templates"].T)
        plt.axvline(x=template_r_position, color="r", label="R peak")
        plt.axvline(x=Q_positions_template[0], color="yellow", label="Q positions")
        for position in range(1, len(Q_positions_template)):
            plt.axvline(
                x=Q_positions_template[position],
                color="yellow",
            )
        plt.axvline(
            x=Q_start_positions_template[0], color="green", label="Q Start positions"
        )
        for position in range(1, len(Q_start_positions_template)):
            plt.axvline(
                x=Q_start_positions_template[position],
                color="green",
            )
        plt.legend()
        plt.show()

        Q_positions = np.array(Q_positions)
        Q_start_positions = np.array(Q_start_positions)

    return utils.ReturnTuple(
        (
            Q_positions,
            Q_start_positions,
        ),
        (
            "Q_positions",
            "Q_start_positions",
        ),
    )


def getSPositions(ecg_proc=None, show=False):
    """Different ECG Waves (Q, R, S, ...) are not present or are not so clear to identify in all ECG signals (I II III V1 V2 V3, ...)
       For S wave we suggest to use signals V1, V2, V3. Avoid I, V5, V6, aVR, aVL

    Parameters
    ----------
    signal : object
    object return by the function ecg.
    show : bool, optional
    If True, show a plot of the S Positions on every signal sample/template.

    Returns
    -------
    S_positions : array
            Array with all S positions on the signal
    S_end_ positions : array
            Array with all S end positions on the signal
    """

    templates_ts = ecg_proc["templates_ts"]
    template_r_position = np.argmin(
        np.abs(templates_ts - 0)
    )  # R peak on the template is always on time instant 0 seconds
    S_positions = []
    S_end_positions = []
    S_positions_template = []
    S_end_positions_template = []
    template_size = len(ecg_proc["templates"][0])

    for n, each in enumerate(ecg_proc["templates"]):
        # Get S Position
        template_right = each[template_r_position : template_size + 1]
        mininums_from_template_right = argrelextrema(template_right, np.less)

        try:
            S_position = ecg_proc["rpeaks"][n] + mininums_from_template_right[0][0]
            S_positions.append(S_position)
            S_positions_template.append(
                template_r_position + mininums_from_template_right[0][0]
            )
        except:
            pass
        # Get S end position
        maximums_from_template_right = argrelextrema(template_right, np.greater)
        try:
            S_end_position = ecg_proc["rpeaks"][n] + maximums_from_template_right[0][0]
            S_end_positions.append(S_end_position)
            S_end_positions_template.append(
                template_r_position + maximums_from_template_right[0][0]
            )
        except:
            pass

    if show:
        plt.figure()
        plt.plot(ecg_proc["templates"].T)
        plt.axvline(x=template_r_position, color="r", label="R peak")
        plt.axvline(x=S_positions_template[0], color="yellow", label="S positions")
        for position in range(1, len(S_positions_template)):
            plt.axvline(
                x=S_positions_template[position],
                color="yellow",
            )

        plt.axvline(
            x=S_end_positions_template[0], color="green", label="S end positions"
        )
        for position in range(1, len(S_end_positions_template)):
            plt.axvline(
                x=S_end_positions_template[position],
                color="green",
            )

        plt.legend()
        plt.show()

        S_positions = np.array(S_positions)
        S_end_positions = np.array(S_end_positions)

    return utils.ReturnTuple(
        (
            S_positions,
            S_end_positions,
        ),
        (
            "S_positions",
            "S_end_positions",
        ),
    )


def getPPositions(ecg_proc=None, show=False):
    """Different ECG Waves (Q, R, S, ...) are not present or are not so clear to identify in all ECG signals (I II III V1 V2 V3, ...)
       For P wave we suggest to use signals II, V1, aVF . Avoid I, III, V1, V2, V3, V4, V5, AVL

    Parameters
    ----------
    signal : object
    object return by the function ecg.
    show : bool, optional
    If True, show a plot of the P Positions on every signal sample/template.

    Returns
    -------
    P_positions : array
            Array with all P positions on the signal
    P_start_ positions : array
            Array with all P start positions on the signal
    P_end_ positions : array
            Array with all P end positions on the signal
    """

    templates_ts = ecg_proc["templates_ts"]
    # R peak on the template is always on time instant 0 seconds
    template_r_position = np.argmin(np.abs(templates_ts - 0))
    # the P wave end is approximately 0.04 seconds before the R peak
    template_p_position_max = np.argmin(np.abs(templates_ts - (-0.04)))

    P_positions = []
    P_start_positions = []
    P_end_positions = []

    P_positions_template = []
    P_start_positions_template = []
    P_end_positions_template = []

    for n, each in enumerate(ecg_proc["templates"]):
        # Get P position
        template_left = each[0 : template_p_position_max + 1]
        max_from_template_left = np.argmax(template_left)
        # print("P Position=" + str(max_from_template_left))
        P_position = (
            ecg_proc["rpeaks"][n] - template_r_position + max_from_template_left
        )
        P_positions.append(P_position)
        P_positions_template.append(max_from_template_left)

        # Get P start position
        template_P_left = each[0 : max_from_template_left + 1]
        mininums_from_template_left = argrelextrema(template_P_left, np.less)
        # print("P start position=" + str(mininums_from_template_left[0][-1]))
        try:
            P_start_position = (
                ecg_proc["rpeaks"][n]
                - template_r_position
                + mininums_from_template_left[0][-1]
            )
            P_start_positions.append(P_start_position)
            P_start_positions_template.append(mininums_from_template_left[0][-1])
        except:
            pass

        # Get P end position
        template_P_right = each[max_from_template_left : template_p_position_max + 1]
        mininums_from_template_right = argrelextrema(template_P_right, np.less)

        try:
            P_end_position = (
                ecg_proc["rpeaks"][n]
                - template_r_position
                + max_from_template_left
                + mininums_from_template_right[0][0]
            )

            P_end_positions.append(P_end_position)
            P_end_positions_template.append(
                max_from_template_left + mininums_from_template_right[0][0]
            )
        except:
            pass

    if show:
        plt.figure()
        plt.plot(ecg_proc["templates"].T)
        plt.axvline(x=template_r_position, color="r", label="R peak")
        plt.axvline(x=P_positions_template[0], color="yellow", label="P positions")
        for position in range(1, len(P_positions_template)):
            plt.axvline(
                x=P_positions_template[position],
                color="yellow",
            )
        plt.axvline(x=P_start_positions_template[0], color="green", label="P starts")
        for position in range(1, len(P_start_positions_template)):
            plt.axvline(
                x=P_start_positions_template[position],
                color="green",
            )
        plt.axvline(x=P_end_positions_template[0], color="green", label="P ends")
        for position in range(1, len(P_end_positions_template)):
            plt.axvline(
                x=P_end_positions_template[position],
                color="green",
            )

        plt.legend()
        plt.show()

        P_positions = np.array(P_positions)
        P_start_positions = np.array(P_start_positions)
        P_end_positions = np.array(P_end_positions)

    return utils.ReturnTuple(
        (
            P_positions,
            P_start_positions,
            P_end_positions,
        ),
        (
            "P_positions",
            "P_start_positions",
            "P_end_positions",
        ),
    )


def getTPositions(ecg_proc=None, show=False):
    """Different ECG Waves (Q, R, S, ...) are not present or are not so clear to identify in all ECG signals (I II III V1 V2 V3, ...)
    For T wave we suggest to use signals V4, v5 (II, V3 have good results, but in less accuracy) . Avoid I, V1, V2, aVR, aVL

    Parameters
    ----------
    signal : object
    object return by the function ecg.
    show : bool, optional
    If True, show a plot of the T Positions on every signal sample/template.

    Returns
    -------
    T_positions : array
        Array with all T positions on the signal
    T_start_ positions : array
        Array with all T start positions on the signal
    T_end_ positions : array
        Array with all T end positions on the signal
    """

    templates_ts = ecg_proc["templates_ts"]

    # R peak on the template is always on time instant 0 seconds
    template_r_position = np.argmin(np.abs(templates_ts - 0))
    # the T wave start is approximately 0.14 seconds after R-peak
    template_T_position_min = np.argmin(np.abs(templates_ts - 0.14))

    T_positions = []
    T_start_positions = []
    T_end_positions = []

    T_positions_template = []
    T_start_positions_template = []
    T_end_positions_template = []

    for n, each in enumerate(ecg_proc["templates"]):
        # Get T position
        template_right = each[template_T_position_min:]
        max_from_template_right = np.argmax(template_right)
        # print("T Position=" + str(template_T_position_min + max_from_template_right))
        T_position = (
            ecg_proc["rpeaks"][n]
            - template_r_position
            + template_T_position_min
            + max_from_template_right
        )

        T_positions.append(T_position)
        T_positions_template.append(template_T_position_min + max_from_template_right)

        # Get T start position
        template_T_left = each[
            template_r_position : template_T_position_min + max_from_template_right
        ]
        min_from_template_T_left = argrelextrema(template_T_left, np.less)

        try:
            T_start_position = ecg_proc["rpeaks"][n] + min_from_template_T_left[0][-1]

            T_start_positions.append(T_start_position)
            T_start_positions_template.append(
                template_r_position + min_from_template_T_left[0][-1]
            )
        except:
            pass

        # Get T end position
        template_T_right = each[template_T_position_min + max_from_template_right :]

        mininums_from_template_T_right = argrelextrema(template_T_right, np.less)

        try:
            T_end_position = (
                ecg_proc["rpeaks"][n]
                - template_r_position
                + template_T_position_min
                + max_from_template_right
                + mininums_from_template_T_right[0][0]
            )

            T_end_positions.append(T_end_position)
            T_end_positions_template.append(
                template_T_position_min
                + max_from_template_right
                + mininums_from_template_T_right[0][0]
            )
        except:
            pass

    if show:
        plt.figure()
        plt.plot(ecg_proc["templates"].T)
        plt.axvline(x=template_r_position, color="r", label="R peak")
        plt.axvline(x=T_positions_template[0], color="yellow", label="T positions")
        for position in range(1, len(T_positions_template)):
            plt.axvline(
                x=T_positions_template[position],
                color="yellow",
            )
        plt.axvline(x=T_start_positions_template[0], color="green", label="T starts")
        for position in range(1, len(T_start_positions_template)):
            plt.axvline(
                x=T_start_positions_template[position],
                color="green",
            )
        plt.axvline(x=T_end_positions_template[0], color="green", label="T ends")
        for position in range(1, len(T_end_positions_template)):
            plt.axvline(
                x=T_end_positions_template[position],
                color="green",
            )
        plt.legend()
        plt.show()

        T_positions = np.array(T_positions)
        T_start_positions = np.array(T_start_positions)
        T_end_positions = np.array(T_end_positions)

    return utils.ReturnTuple(
        (
            T_positions,
            T_start_positions,
            T_end_positions,
        ),
        (
            "T_positions",
            "T_start_positions",
            "T_end_positions",
        ),
    )


def bSQI(detector_1, detector_2, fs=1000.0, mode="simple", search_window=150):
    """Comparison of the output of two detectors.

    Parameters
    ----------
    detector_1 : array
        Output of the first detector.
    detector_2 : array
        Output of the second detector.
    fs: int, optional
        Sampling rate, in Hz.
    mode : str, optional
        If 'simple', return only the percentage of beats detected by both. If 'matching', return the peak matching degree.
        If 'n_double' returns the number of matches divided by the sum of all minus the matches.
    search_window : int, optional
        Search window around each peak, in ms.

    Returns
    -------
    bSQI : float
        Performance of both detectors.

    """

    if detector_1 is None or detector_2 is None:
        raise TypeError("Input Error, check detectors outputs")
    search_window = int(search_window / 1000 * fs)
    both = 0
    for i in detector_1:
        for j in range(max([0, i - search_window]), i + search_window):
            if j in detector_2:
                both += 1
                break

    if mode == "simple":
        return (both / len(detector_1)) * 100
    elif mode == "matching":
        return (2 * both) / (len(detector_1) + len(detector_2))
    elif mode == "n_double":
        return both / (len(detector_1) + len(detector_2) - both)


def sSQI(signal):
    """Return the skewness of the signal

    Parameters
    ----------
    signal : array
        ECG signal.

    Returns
    -------
    skewness : float
        Skewness value.

    """
    if signal is None:
        raise TypeError("Please specify an input signal")

    return stats.skew(signal)


def kSQI(signal, fisher=True):
    """Return the kurtosis of the signal

    Parameters
    ----------
    signal : array
        ECG signal.
    fisher : bool, optional
        If True,Fisher’s definition is used (normal ==> 0.0). If False, Pearson’s definition is used (normal ==> 3.0).

    Returns
    -------
    kurtosis : float
        Kurtosis value.
    """

    if signal is None:
        raise TypeError("Please specify an input signal")

    return stats.kurtosis(signal, fisher=fisher)


def pSQI(signal, f_thr=0.01):
    """Return the flatline percentage of the signal

    Parameters
    ----------
    signal : array
        ECG signal.
    f_thr : float, optional
        Flatline threshold, in mV / sample

    Returns
    -------
    flatline_percentage : float
        Percentage of signal where the absolute value of the derivative is lower then the threshold.

    """

    if signal is None:
        raise TypeError("Please specify an input signal")

    diff = np.diff(signal)
    length = len(diff)

    flatline = np.where(abs(diff) < f_thr)[0]

    return (len(flatline) / length) * 100


def fSQI(
    ecg_signal,
    fs=1000.0,
    nseg=1024,
    num_spectrum=[5, 20],
    dem_spectrum=None,
    mode="simple",
):
    """Returns the ration between two frequency power bands.

    Parameters
    ----------
    ecg_signal : array
        ECG signal.
    fs : float, optional
        ECG sampling frequency, in Hz.
    nseg : int, optional
        Frequency axis resolution.
    num_spectrum : array, optional
        Frequency bandwidth for the ratio's numerator, in Hz.
    dem_spectrum : array, optional
        Frequency bandwidth for the ratio's denominator, in Hz. If None, then the whole spectrum is used.
    mode : str, optional
        If 'simple' just do the ration, if is 'bas', then do 1 - num_power.

    Returns
    -------
    Ratio : float
        Ratio between two powerbands.
    """

    def power_in_range(f_range, f, Pxx_den):
        _indexes = np.where((f >= f_range[0]) & (f <= f_range[1]))[0]
        _power = integrate.trapz(Pxx_den[_indexes], f[_indexes])
        return _power

    if ecg_signal is None:
        raise TypeError("Please specify an input signal")

    f, Pxx_den = ss.welch(ecg_signal, fs, nperseg=nseg)
    num_power = power_in_range(num_spectrum, f, Pxx_den)

    if dem_spectrum is None:
        dem_power = power_in_range([0, float(fs / 2.0)], f, Pxx_den)
    else:
        dem_power = power_in_range(dem_spectrum, f, Pxx_den)

    if mode == "simple":
        return num_power / dem_power
    elif mode == "bas":
        return 1 - num_power / dem_power


def ZZ2018(
    signal, detector_1, detector_2, fs=1000, search_window=100, nseg=1024, mode="simple"
):
    import numpy as np

    """ Signal quality estimator. Designed for signal with a lenght of 10 seconds.
        Follows the approach by Zhao *et la.* [Zhao18]_.

    Parameters
    ----------
    signal : array
        Input ECG signal in mV.
    detector_1 : array
        Input of the first R peak detector.
    detector_2 : array
        Input of the second R peak detector.
    fs : int, float, optional
        Sampling frequency (Hz).
    search_window : int, optional
        Search window around each peak, in ms.
    nseg : int, optional
        Frequency axis resolution.
    mode : str, optional
        If 'simple', simple heurisitc. If 'fuzzy', employ a fuzzy classifier.

    Returns
    -------
    noise : str
        Quality classification.

    References
    ----------
    .. [Zhao18] Zhao, Z., & Zhang, Y. (2018).
    SQI quality evaluation mechanism of single-lead ECG signal based on simple heuristic fusion and fuzzy comprehensive evaluation.
    Frontiers in Physiology, 9, 727.
    """

    if len(detector_1) == 0 or len(detector_2) == 0:
        return "Unacceptable"

    ## compute indexes
    qsqi = bSQI(
        detector_1, detector_2, fs=fs, mode="matching", search_window=search_window
    )
    psqi = fSQI(signal, fs=fs, nseg=nseg, num_spectrum=[5, 15], dem_spectrum=[5, 40])
    ksqi = kSQI(signal)
    bassqi = fSQI(
        signal, fs=fs, nseg=nseg, num_spectrum=[0, 1], dem_spectrum=[0, 40], mode="bas"
    )

    if mode == "simple":
        ## First stage rules (0 = unqualified, 1 = suspicious, 2 = optimal)
        ## qSQI rules
        if qsqi > 0.90:
            qsqi_class = 2
        elif qsqi < 0.60:
            qsqi_class = 0
        else:
            qsqi_class = 1

        ## pSQI rules
        import numpy as np

        ## Get the maximum bpm
        if len(detector_1) > 1:
            RR_max = 60000.0 / (1000.0 / fs * np.min(np.diff(detector_1)))
        else:
            RR_max = 1

        if RR_max < 130:
            l1, l2, l3 = 0.5, 0.8, 0.4
        else:
            l1, l2, l3 = 0.4, 0.7, 0.3

        if psqi > l1 and psqi < l2:
            pSQI_class = 2
        elif psqi > l3 and psqi < l1:
            pSQI_class = 1
        else:
            pSQI_class = 0

        ## kSQI rules
        if ksqi > 5:
            kSQI_class = 2
        else:
            kSQI_class = 0

        ## basSQI rules
        if bassqi >= 0.95:
            basSQI_class = 2
        elif bassqi < 0.9:
            basSQI_class = 0
        else:
            basSQI_class = 1

        class_matrix = np.array([qsqi_class, pSQI_class, kSQI_class, basSQI_class])
        n_optimal = len(np.where(class_matrix == 2)[0])
        n_suspics = len(np.where(class_matrix == 1)[0])
        n_unqualy = len(np.where(class_matrix == 0)[0])
        if (
            n_unqualy >= 3
            or (n_unqualy == 2 and n_suspics >= 1)
            or (n_unqualy == 1 and n_suspics == 3)
        ):
            return "Unacceptable"
        elif n_optimal >= 3 and n_unqualy == 0:
            return "Excellent"
        else:
            return "Barely acceptable"

    elif mode == "fuzzy":
        # Transform qSQI range from [0, 1] to [0, 100]
        qsqi = qsqi * 100.0
        # UqH (Excellent)
        if qsqi <= 80:
            UqH = 0
        elif qsqi >= 90:
            UqH = qsqi / 100.0
        else:
            UqH = 1.0 / (1 + (1 / np.power(0.3 * (qsqi - 80), 2)))

        # UqI (Barely acceptable)
        UqI = 1.0 / (1 + np.power((qsqi - 75) / 7.5, 2))

        # UqJ (unacceptable)
        if qsqi <= 55:
            UqJ = 1
        else:
            UqJ = 1.0 / (1 + np.power((qsqi - 55) / 5.0, 2))

        # Get R1
        R1 = np.array([UqH, UqI, UqJ])

        # pSQI
        # UpH
        if psqi <= 0.25:
            UpH = 0
        elif psqi >= 0.35:
            UpH = 1
        else:
            UpH = 0.1 * (psqi - 0.25)

        # UpI
        if psqi < 0.18:
            UpI = 0
        elif psqi >= 0.32:
            UpI = 0
        elif psqi >= 0.18 and psqi < 0.22:
            UpI = 25 * (psqi - 0.18)
        elif psqi >= 0.22 and psqi < 0.28:
            UpI = 1
        else:
            UpI = 25 * (0.32 - psqi)

        # UpJ
        if psqi < 0.15:
            UpJ = 1
        elif psqi > 0.25:
            UpJ = 0
        else:
            UpJ = 0.1 * (0.25 - psqi)

        # Get R2
        R2 = np.array([UpH, UpI, UpJ])

        # kSQI
        # Get R3
        if ksqi > 5:
            R3 = np.array([1, 0, 0])
        else:
            R3 = np.array([0, 0, 1])

        # basSQI
        # UbH
        if bassqi <= 90:
            UbH = 0
        elif bassqi >= 95:
            UbH = bassqi / 100.0
        else:
            UbH = 1.0 / (1 + (1 / np.power(0.8718 * (bassqi - 90), 2)))

        # UbI
        if bassqi <= 85:
            UbI = 1
        else:
            UbI = 1.0 / (1 + np.power((bassqi - 85) / 5.0, 2))

        # UbJ
        UbJ = 1.0 / (1 + np.power((bassqi - 95) / 2.5, 2))

        # R4
        R4 = np.array([UbH, UbI, UbJ])

        # evaluation matrix R
        R = np.vstack([R1, R2, R3, R4])

        # weight vector W
        W = np.array([0.4, 0.4, 0.1, 0.1])

        S = np.array(
            [np.sum((R[:, 0] * W)), np.sum((R[:, 1] * W)), np.sum((R[:, 2] * W))]
        )

        # classify
        V = np.sum(np.power(S, 2) * [1, 2, 3]) / np.sum(np.power(S, 2))

        if V < 1.5:
            return "Excellent"
        elif V >= 2.40:
            return "Unnacceptable"
        else:
            return "Barely acceptable"
