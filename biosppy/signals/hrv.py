# -*- coding: utf-8 -*-
"""
biosppy.signals.hrv
-------------------

This module provides computation and visualization of Heart-Rate Variability
metrics.


:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
# compat
from __future__ import absolute_import, division, print_function

# 3rd party
import numpy as np
import warnings

# local
from .. import utils


def compute_rri(rpeaks, sampling_rate=1000.):
    """ Computes RR intervals in milliseconds from a list of R-peak indexes.

    Parameters
    ----------
    rpeaks : list, array
        R-peak index locations.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    rri : array
        RR-intervals (ms).
    """

    # ensure input format
    rpeaks = np.array(rpeaks)

    # difference of R-peaks converted to ms
    rri = (1000. * np.diff(rpeaks)) / sampling_rate

    # check if rri is within physiological parameters
    if rri.min() < 400 or rri.min() > 1400:
        warnings.warn("RR-intervals appear to be out of normal parameters. Check input values.")

    return rri


def filter_rri(rri=None, threshold=1200):
    """Filters an RRI sequence based on a maximum threshold in milliseconds.

    Parameters
    ----------
    rri : array
        RR-intervals (default: ms).
    threshold : int, float, optional
        Maximum rri value to accept (ms).
    """

    # ensure input format
    rri = np.array(rri, dtype=float)

    # filter rri values
    rri_filt = rri[np.where(rri < threshold)]

    return rri_filt


def hrv_timedomain(rri, duration=None):
    """ Computes the time domain HRV features from a sequence of RR intervals in milliseconds.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).

    Returns
    -------
    hr : array
        Heart rate (bpm).
    hr_min : float
        Minimum heart rate (bpm).
    hr_max : float
        Maximum heart rate (bpm).
    hr_minmax :  float
        Difference between the highest and the lowest heart rate (bpm).
    hr_avg : float
        Average heart rate (bpm).
    rmssd : float
        RMSSD - Root mean square of successive RR interval differences (ms).
    nn50 : int
        NN50 - Number of successive RR intervals that differ by more than 50ms.
    pnn50 : float
        pNN50 - Percentage of successive RR intervals that differ by more than
        50ms.
    sdnn: float
       SDNN - Standard deviation of RR intervals (ms).
    """

    # check inputs
    if rri is None:
        raise ValueError("Please specify an RRI list or array.")

    # ensure numpy
    rri = np.array(rri, dtype=float)

    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 10:
        raise ValueError("Signal must be longer than 10 seconds to compute time-domain features.")

    # initialize outputs
    out = utils.ReturnTuple((), ())

    # compute the difference between RRIs
    rri_diff = np.diff(rri)

    if duration >= 10:
        # compute heart rate features
        hr = 60/(rri/1000)  # bpm
        hr_min = hr.min()
        hr_max = hr.max()
        hr_minmax = hr.max() - hr.min()
        hr_avg = hr.mean()

        out = out.append([hr, hr_min, hr_max, hr_minmax, hr_avg],
                         ['hr', 'hr_min', 'hr_max', 'hr_minmax', 'hr_avg'])

        # compute RMSSD
        rmssd = (rri_diff ** 2).mean() ** 0.5

        out = out.append(rmssd, 'rmssd')

    if duration >= 20:
        # compute NN50 and pNN50
        th50 = 50
        nntot = len(rri_diff)
        nn50 = len(np.argwhere(abs(rri_diff) > th50))
        pnn50 = 100 * (nn50 / nntot)

        out = out.append([nn50, pnn50], ['nn50', 'pnn50'])

    if duration >= 60:
        # compute SDNN
        sdnn = rri.std()

        out = out.append(sdnn, 'sdnn')

    return out

