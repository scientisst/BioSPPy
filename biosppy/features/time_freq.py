# -*- coding: utf-8 -*-
"""
biosppy.features.time_freq
--------------------------

This module provides methods to extract time-frequency features using
discrete wavelet decomposition.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# 3rd party
import numpy as np
import pywt

# local
from .. import utils
from ..signals import tools as st


def time_freq(signal=None, wavelet="db4", level=5):
    """Compute statistical metrics over the signal discrete wavelet transform
    approximation and detail coefficients.

    Parameters
    ----------
    signal : array
        Input signal.
    wavelet: str
        Type of wavelet. Default is db4 (Daubechies 4).
    level: int
        Decomposition level. Default is 5.

    Returns
    -------
    dwt_app_{metric} : float
        Statistical metrics over the approximation coefficients.
    dwt_det{level}_{metric} : float
        Statistical metrics over the detail coefficients at the specified level.

    Notes
    -----
    Check biosppy.signals.tools.signal_stats for the list of available metrics.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # initialize output
    feats = utils.ReturnTuple((), ())

    # compute wavelet
    coeffs = compute_wavelet(signal=signal, wavelet=wavelet, level=level)

    # compute stats
    for coeff, coeff_name in zip(coeffs, coeffs.keys()):
        signal_feats = st.signal_stats(coeff)
        for stat_value, stat_name in zip(signal_feats, signal_feats.keys()):
            feats = feats.append(stat_value, coeff_name + '_' + stat_name)

    return feats


def compute_wavelet(signal=None, wavelet="db4", level=5):
    """Compute the approximation and highest detail coefficients of the signal
    using the discrete wavelet transform.

    Parameters
    ----------
    signal : array
        Input signal.
    wavelet: str
        Type of wavelet.
    level: int
        Decomposition level

    Returns
    -------
    dwt_app : array
        Approximation coefficients.
    dwt_det{level} : array
        Detail coefficients at the specified level.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # wavelets
    dwt_det = pywt.downcoef("d", signal, wavelet, level)
    dwt_app = pywt.downcoef("a", signal, wavelet, level)

    # output
    args = (dwt_app, dwt_det)
    names = ("dwt_app", f"dwt_det{level}")

    return utils.ReturnTuple(args, names)
