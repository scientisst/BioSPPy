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


def time_feats(signal=None, sampling_rate=1000.):
    """Compute various time metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : int, float, optional
        Sampling Rate (Hz).

    Returns
    -------
    max : float
        Signal maximum amplitude.
    min: float
        Signal minimum amplitude.
    range: float
        Signal range amplitude.
    iqr : float
        Interquartile range.
    mean : float
        Signal average.
    std : float
        Signal standard deviation.
    max_to_mean: float
        Signal maximum amplitude to mean.
    dist : float
        Length of the signal (sum of abs diff).
    mean_AD1 : float
        Mean absolute differences.
    med_AD1 : float
        Median absolute differences.
    min_AD1 : float
        Min absolute differences.
    max_AD1 : float
        Maximum absolute differences.
    mean_D1 : float
        Mean of differences.
    me_dD1 : float
        Median of differences.
    std_D1 : float
        Standard deviation of differences.
    max_D1 : float
        Max of differences.
    min_D1 : float
        Min of differences.
    sum_D1 : float
        Sum of differences.
    range_D1 : float
        Amplitude range of differences.
    iqr_d1 : float
        Interquartile range of differences.
    mean_D2 : float
        Mean of 2nd differences.
    std_D2 : float
        Standard deviation of 2nd differences.
    max_D2 : float
        Max of 2nd differences.
    min_D2 : float
        Min of 2nd differences.
    sum_D2: float
        Sum of 2nd differences.
    range_D2 : float
        Amplitude range of 2nd differences.
    iqr_D2 : float
        Interquartile range of 2nd differences.
    autocorr : float
        Signal autocorrelation sum.
    zero_cross : int
        Number of times the signal crosses the zero axis.
    min_peaks : int
        Number of minimum peaks.
    max_peaks : int
        Number of maximum peaks.   
    total_e : float
        Total energy.
    lin_reg_slope : float
        Slope of linear regression. 
    lin_reg_b : float
        Interception coefficient b of linear regression. 
    corr_lin_reg: float
        Correlation to linear regression.
    mobility: float
        ratio of the variance between the first derivative and the signal.
    complexity: float
        ratio between the mobility of the derivative and the mobility of the signal.
    chaos: float
        ratio between the complexity of the derivative and the complexity of the signal.
    hazard: float
        ratio between the chaos of the derivative and the chaos of the signal.
    kurtosis : float
        Signal kurtosis (unbiased).
    skewness : float
        Signal skewness (unbiased).
    rms : float
        Root Mean Square.
    midhinge: float
        average of first and third quartile.
    trimean: float
        weighted average of 1st, 2nd and 3rd quartiles.
    stat_hist{bins} : float
        Relative frequency of 5 histogram bins.
    entropy : float
        Signal entropy.
    
    References
    ----------
    - TSFEL library: https://github.com/fraunhoferportugal/tsfel
    - Peeters, Geoffroy. (2004). A large set of audio features for sound description (similarity and classification) in the CUIDADO project.
    - Veeranki, Yedukondala Rao, Nagarajan Ganapathy, and Ramakrishnan Swaminathan. "Non-Parametric Classifiers Based Emotion Classification Using Electrodermal Activity and Modified Hjorth Features." MIE. 2021.
    - Ghaderyan, Peyvand, and Ataollah Abbasi. "An efficient automatic workload estimation method based on electrodermal activity using pattern classifier combinations." International Journal of Psychophysiology 110 (2016): 91-101.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # helpers
    args, names = [], []
    sig_diff = np.diff(signal)  # 1st derivative
    sig_diff_2 = np.diff(sig_diff)  # 2nd derivative
    mean = np.mean(signal)
    time = range(len(signal))
    time = [float(x) / sampling_rate for x in time]
    ds = 1/sampling_rate
    energy = np.sum(signal**2*ds)

    # max to mean ❌
    max_to_mean = np.max(signal - mean)
    args += [max_to_mean]
    names += ['max_to_mean']

    # distance
    dist = np.sum([1 if d == 0 else d for d in np.abs(sig_diff)]) + 1
    args += [dist]
    names += ['dist']

    # autocorrelation sum
    if np.sum(np.abs(signal)) > 0:
        autocorr = np.sum(np.correlate(signal, signal, 'full'))
    else:
        autocorr = None
    args += [autocorr]
    names += ['autocorr']

    # zero_cross
    zero_cross = len(np.where(np.abs(np.diff(np.sign(signal))) >= 1)[0])
    args += [zero_cross]
    names += ['zero_cross']

    # number of minimum peaks
    min_peaks = len(tools.find_extrema(signal, "min")["extrema"])
    args += [min_peaks]
    names += ['min_peaks']

    # number of maximum peaks
    max_peaks = len(tools.find_extrema(signal, "max")["extrema"])
    args += [max_peaks]
    names += ['max_peaks']
    
    # total energy
    total_e = np.sum(energy)
    args += [total_e]
    names += ['total_e']

    _t = np.array(time).reshape(-1, 1)
    reg = linear_model.LinearRegression().fit(_t,  signal)
    lin_reg_slope = reg.coef_[0]
    args += [lin_reg_slope]
    names += ['lin_reg_slope']

    lin_reg_b = reg.intercept_
    args += [lin_reg_b]
    names += ['lin_reg_b']

    c = 0
    if np.sum(np.abs(signal)) > 0 and len(np.unique(signal)) > 1:
        _r = reg.predict(_t)
        if np.sum(np.abs(_r)) > 0 and len(np.unique(_r)) > 1:
            corr_lin_reg = pearson_correlation(signal, reg.predict(_t))[0]
            c = 1
    if not c:
        corr_lin_reg = None

    args += [corr_lin_reg]
    names += ['corr_lin_reg']

    # entropy
    if np.sum(np.abs(signal)) > 0:
        _entropy = np.nan_to_num(entropy(signal))
    else:
        _entropy = None
    args += [_entropy]
    names += ['entropy']

    # output
    args = tuple(args)
    names = tuple(names)

    return utils.ReturnTuple(args, names)


def hjorth_mob(signal=None):
    """Compute signal mobility hjorth feature, that is, the ratio of the
    variance between the first derivative and the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    mobility : float
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
    names = ('mobility',)

    return utils.ReturnTuple(args, names)


def hjorth_comp(signal=None):
    """Compute signal complexity hjorth feature, that is, the ratio between the
     mobility of the derivative and the mobility of the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    complexity : float
        Signal complexity.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    d = np.diff(signal)

    try:
        mob_signal = hjorth_mob(signal)["mobility"]
        mob_d = hjorth_mob(d)["mobility"]
        complexity = mob_d / mob_signal

    except Exception as e:
        print("complexity", e)
        complexity = None

    # output
    args = (complexity,)
    names = ('complexity',)

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
    chaos : float
        Signal chaos.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    d = np.diff(signal)

    try:
        comp_signal = hjorth_comp(signal)['complexity']
        comp_d = hjorth_comp(d)['complexity']
        chaos = comp_d / comp_signal

    except Exception as e:
        print("chaos", e)
        chaos = None

    # output
    args = (chaos,)
    names = ('chaos',)

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
    hazard : float
        Signal hazard.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    d = np.diff(signal)

    if hjorth_chaos(signal)['chaos'] is not None:
        hazard = hjorth_chaos(d)['chaos'] / hjorth_chaos(signal)['chaos']
    else:
        hazard = None

    # output
    args = (hazard,)
    names = ('hazard',)

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
