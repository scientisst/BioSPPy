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
        - biosppy.stats.histogram

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

    # quartile features
    # iqr
    q1, q3 = signal_feats['q1'], signal_feats['q3']
    iqr = q3 - q1
    feats = feats.append(iqr, 'iqr')

    # midhinge
    midhinge = (q3 + q1) / 2
    feats = feats.append(midhinge, 'midhinge')

    # trimean
    trimean = (signal_feats['median'] + midhinge) / 2
    feats = feats.append(trimean, 'trimean')

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

    # hjorth mobility
    mobility = hjorth_mobility(signal)
    feats = feats.join(mobility)

    # hjorth complexity
    complexity = hjorth_complexity(signal)
    feats = feats.join(complexity)

    # hjorth chaos
    chaos = hjorth_chaos(signal)
    feats = feats.join(chaos)

    # hjorth hazard
    hazard = hjorth_hazard(signal)
    feats = feats.join(hazard)

    # diff stats
    if include_diff:
        diff_feats = stats.diff_stats(signal, stats_only=True)
        feats = feats.join(diff_feats)

    return feats


def hjorth_mobility(signal=None):
    """Compute signal mobility hjorth feature, that is, the ratio of the
    variance between the first derivative and the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    hjorth_mobility : float
        Signal mobility.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    d = np.diff(signal)

    try:
        if np.var(signal) > 0:
            mobility = np.sqrt(np.var(d)/np.var(signal))
        else:
            mobility = None
    except Exception as e:
        print("mobility", e)
        mobility = None

    # output
    args = (mobility,)
    names = ('hjorth_mobility',)

    return utils.ReturnTuple(args, names)


def hjorth_complexity(signal=None):
    """Compute signal complexity hjorth feature, that is, the ratio between the
     mobility of the derivative and the mobility of the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    hjorth_complexity : float
        Signal complexity.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    d = np.diff(signal)

    try:
        mob_signal = hjorth_mobility(signal)["hjorth_mobility"]
        mob_d = hjorth_mobility(d)["hjorth_mobility"]
        complexity = mob_d / mob_signal

    except Exception as e:
        print("complexity", e)
        complexity = None

    # output
    args = (complexity,)
    names = ('hjorth_complexity',)

    return utils.ReturnTuple(args, names)


def hjorth_chaos(signal=None):
    """Compute signal chaos hjorth feature, that is, the ratio between the
    complexity of the derivative and the complexity of the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    hjorth_chaos : float
        Signal chaos.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    d = np.diff(signal)

    try:
        comp_signal = hjorth_complexity(signal)['hjorth_complexity']
        comp_d = hjorth_complexity(d)['hjorth_complexity']
        chaos = comp_d / comp_signal

    except Exception as e:
        print("chaos", e)
        chaos = None

    # output
    args = (chaos,)
    names = ('hjorth_chaos',)

    return utils.ReturnTuple(args, names)


def hjorth_hazard(signal=None):
    """Compute signal hazard hjorth feature, that is, the ratio between the
    chaos of the derivative and the chaos of the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    hjorth_hazard : float
        Signal hazard.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    d = np.diff(signal)

    if hjorth_chaos(signal)['hjorth_chaos'] is not None:
        hazard = hjorth_chaos(d)['hjorth_chaos'] / hjorth_chaos(signal)['hjorth_chaos']
    else:
        hazard = None

    # output
    args = (hazard,)
    names = ('hjorth_hazard',)

    return utils.ReturnTuple(args, names)
