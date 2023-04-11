# -*- coding: utf-8 -*-
"""
biosppy.features.time
---------------------

This module provides methods to extract time features.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# 3rd party
import numpy as np

# local
from .. import utils
from ..signals import tools as st
from .. import stats


def time(signal=None, sampling_rate=1000., include_diff=True):
    """Compute various time metrics describing the signal.

        Parameters
        ----------
        signal : array
            Input signal.
        sampling_rate : int, float, optional
            Sampling Rate (Hz).
        include_diff : bool, optional
            Whether to include the features of the signal's differences (first, second and absolute).

        Returns
        -------
        feats : ReturnTuple object
            Time features of the signal.

        Notes
        -----
        Besides the features directly extracted in this function, it also calls:
        - biosppy.signals.tools.signal_stats
        - biosppy.stats.quartiles
        - biosppy.stats.histogram
        - biosppy.features.time.hjorth_features

        """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # initialize output
    feats = utils.ReturnTuple((), ())

    # basic stats
    signal_feats = st.signal_stats(signal)
    feats = feats.join(signal_feats)

    # quartile features
    quartile_feats = stats.quartiles(signal)
    feats = feats.join(quartile_feats)

    # number of maxima
    nb_maxima = st.find_extrema(signal, mode="max")
    feats = feats.append(len(nb_maxima['extrema']), 'nb_maxima')

    # number of minima
    nb_minima = st.find_extrema(signal, mode="min")
    feats = feats.append(len(nb_minima['extrema']), 'nb_minima')

    # autocorrelation sum
    autocorr_sum = np.sum(np.correlate(signal, signal, 'full'))
    feats = feats.append(autocorr_sum, 'autocorr_sum')

    # total energy
    total_energy = np.sum(np.abs(signal)**2)
    feats = feats.append(total_energy, 'total_energy')

    # histogram relative frequency
    hist_feats = stats.histogram(signal, normalize=True)
    feats = feats.join(hist_feats)

    # linear regression
    t_signal = np.arange(0, len(signal)) / sampling_rate
    linreg = stats.linear_regression(t_signal, signal, show=False)
    feats = feats.append(linreg['m'], 'linreg_slope')
    feats = feats.append(linreg['b'], 'linreg_intercept')

    # pearson correlation from linear regression
    linreg_pred = linreg['m'] * t_signal + linreg['b']
    pearson_feats = stats.pearson_correlation(signal, linreg_pred)
    feats = feats.append(pearson_feats['r'], 'pearson_r')

    # hjorth features
    hjorth_feats = hjorth_features(signal)
    feats = feats.join(hjorth_feats)

    # diff stats
    if include_diff:
        diff_feats = stats.diff_stats(signal, stats_only=True)
        feats = feats.join(diff_feats)

    return feats


def hjorth_features(signal=None):
    """Compute Hjorth mobility, complexity, chaos and hazard.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    hjorth_mobility : float
        Hjorth mobility.
    hjorth_complexity : float
        Hjorth complexity.
    hjorth_chaos : float
        Hjorth chaos.
    hjorth_hazard : float
        Hjorth hazard.

    Notes
    -----
    Hjorth activity corresponds to the variance of the signal.

    """

    # helper functions
    def _hjorth_mobility(s):
        if np.var(s) == 0:
            return None
        return np.sqrt(np.var(np.diff(s)) / np.var(s))

    def _hjorth_complexity(s):
        mobility = _hjorth_mobility(s)
        if mobility is None or mobility == 0:
            return None
        return _hjorth_mobility(np.diff(s)) / mobility

    def _hjorth_chaos(s):
        complexity = _hjorth_complexity(s)
        if complexity is None or complexity == 0:
            return None
        return _hjorth_complexity(np.diff(s)) / complexity

    def _hjorth_hazard(s):
        chaos = _hjorth_chaos(s)
        if chaos is None or chaos == 0:
            return None
        return _hjorth_chaos(np.diff(s)) / chaos

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # initialize output
    feats = utils.ReturnTuple((), ())

    # hjorth mobility
    signal_mobility = _hjorth_mobility(signal)
    feats = feats.append(signal_mobility, 'hjorth_mobility')
    if signal_mobility is None:
        print("Hjorth mobility is undefined. Returning None.")

    # hjorth complexity
    signal_complexity = _hjorth_complexity(signal)
    feats = feats.append(signal_complexity, 'hjorth_complexity')
    if signal_complexity is None:
        print("Hjorth complexity is undefined. Returning None.")

    # hjorth chaos
    signal_chaos = _hjorth_chaos(signal)
    feats = feats.append(signal_chaos, 'hjorth_chaos')
    if signal_chaos is None:
        print("Hjorth chaos is undefined. Returning None.")

    # hjorth hazard
    signal_hazard = _hjorth_hazard(signal)
    feats = feats.append(signal_hazard, 'hjorth_hazard')
    if signal_hazard is None:
        print("Hjorth hazard is undefined. Returning None.")

    return feats
