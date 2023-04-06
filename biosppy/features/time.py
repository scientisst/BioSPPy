# -*- coding: utf-8 -*-
"""
biosppy.features.time
---------------------

This module provides methods to extract time features.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# 3rd party
import numpy as np
from sklearn import linear_model 
from scipy.stats import iqr, stats, entropy

# local
from .. import utils
from ..signals import tools
from ..stats import pearson_correlation
from . import statistical


def time(signal=None, sampling_rate=1000.):
    """Compute various time metrics describing the signal.

        Parameters
        ----------
        signal : array
            Input signal.
        sampling_rate : int, float, optional
            Sampling Rate (Hz).

        Returns
        -------
        feats : ReturnTuple object
            Signal time features.

        """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # initialize output
    feats = utils.ReturnTuple((), ())

    # helpers
    ts = np.arange(0, len(signal)) / sampling_rate
    energy = np.sum(signal ** 2)

    # maximum amplitude
    max_amp = np.abs(signal - np.mean(signal)).max()
    feats = feats.append(max_amp, 'maxamp')

    # autocorrelation sum
    autocorr_sum = np.sum(np.correlate(signal, signal, 'full'))
    feats = feats.append(autocorr_sum, 'autocorrsum')

    # zero_cross
    zero_cross = len(np.where(np.abs(np.diff(np.sign(signal))) >= 1)[0])
    feats = feats.append(zero_cross, 'zerocross')

    # number of minimum peaks
    min_peaks = len(tools.find_extrema(signal, "min")["extrema"])
    feats = feats.append(min_peaks, 'minpeaks')

    # number of maximum peaks
    max_peaks = len(tools.find_extrema(signal, "max")["extrema"])
    feats = feats.append(max_peaks, 'maxpeaks')

    # total energy
    total_energy = np.sum(energy)
    feats = feats.append(total_energy, 'totalenergy')

    # hjorth mobility
    mobility = hjorth_mob(signal)['mobility']
    feats = feats.append(mobility, 'mobility')

    # hjorth complexity
    complexity = hjorth_comp(signal)['complexity']
    feats = feats.append(complexity, 'complexity')

    # hjorth chaos
    chaos = hjorth_chaos(signal)['chaos']
    feats = feats.append(chaos, 'chaos')

    # hazard
    hazard = hjorth_hazard(signal)['hazard']
    feats = feats.append(hazard, 'hazard')

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


def diff_stats(signal=None):
    """Compute statistical features from the first signal differences, second
    signal differences and absolute signal differences.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    mean_{diff} : float
        Mean of the difference signal.
    median_{diff} : float
        Median of the difference signal.
    min_{diff} : float
        Minimum of the difference signal.
    max_{diff} : float
        Maximum of the difference signal.
    maxamp_{diff} : float
        Maximum amplitude of the difference signal.
    range_{diff} : float
        Range of the difference signal.
    var_{diff} : float
        Variance of the difference signal.
    std_{diff} : float
        Standard deviation of the difference signal.
    sum_{diff} : float
        Sum of the difference signal.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # initialize output
    feats = utils.ReturnTuple((), ())

    # compute differences
    sig_diff = np.diff(signal)
    sig_diff_2 = np.diff(sig_diff)
    sig_diff_abs = np.abs(sig_diff)

    diffs = [sig_diff, sig_diff_2, sig_diff_abs]
    labels = ['firstdiff', 'seconddiff', 'absdiff']

    # extract features
    for diff, label in zip(diffs, labels):
        # mean
        mean = np.mean(diff)
        feats = feats.append(mean, label + '_' + 'mean')

        # median
        median = np.median(diff)
        feats = feats.append(median, label + '_' + 'median')

        # min
        min_ = diff.min()
        feats = feats.append(min_, label + '_' + 'min')

        # max
        max_ = diff.max()
        feats = feats.append(max_, label + '_' + 'max')

        # maximum amplitude
        max_amp = np.abs(diff - mean).max()
        feats = feats.append(max_amp, label + '_' + 'maxamp')

        # range
        range_ = diff.max() - diff.min()
        feats = feats.append(range_, label + '_' + 'range')

        # variance
        var_ = diff.var(ddof=1)
        feats = feats.append(var_, label + '_' + 'var')

        # standard deviation
        std_ = diff.std(ddof=1)
        feats = feats.append(std_, label + '_' + 'std')

        # sum
        sum_ = np.sum(diff)
        feats = feats.append(sum_, label + '_' + 'sum')

    return feats


def linear_regression_feats(signal=None, sampling_rate=1000.):

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # initialize output
    feats = utils.ReturnTuple((), ())

    # compute linear regression
    ts = np.arange(0, len(signal)) / sampling_rate
    reg = linear_model.LinearRegression().fit(ts,  signal)

    # slope
    lin_reg_slope = reg.coef_[0]
    feats = feats.append(lin_reg_slope, 'lin_reg_slope')

    # intercept
    lin_reg_b = reg.intercept_
    feats = feats.append(lin_reg_b, 'lin_reg_b')

    # pearson correlation
    c = 0
    if np.sum(np.abs(signal)) > 0 and len(np.unique(signal)) > 1:
        _r = reg.predict(ts)
        if np.sum(np.abs(_r)) > 0 and len(np.unique(_r)) > 1:
            corr_lin_reg = pearson_correlation(signal, reg.predict(ts))[0]
            c = 1
    if not c:
        corr_lin_reg = None
    feats = feats.append(corr_lin_reg, 'corr_lin_reg')

    return feats
