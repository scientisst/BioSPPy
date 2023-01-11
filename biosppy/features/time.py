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


def hjorth_mob(signal):
    """Compute signal mobility hjorth feature.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    mobility : float
        Signal mobility.

    """

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


def hjorth_comp(signal):
    """Compute signal complexity hjorth feature.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    complexity : float
        Signal complexity. 

    """

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


def hjorth_chaos(signal):
    """Compute signal chaos hjorth feature.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    chaos : float
        Signal chaos. 

    """

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


def time_features(signal, sampling_rate):
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
   
    # start helpers
    args, names = [], []
    try:
        sig_diff = np.diff(signal)
    except Exception as e:
        print("sig_diff", e)
        sig_diff = []
    
    # 2nd derivative
    try:
        sig_diff_2 = np.diff(sig_diff)
    except Exception as e:
        print("sig_diff_2", e)
        sig_diff_2 = []

    try:
        mean = np.mean(signal)
    except Exception as e:
        print("mean", e)
        mean = None

    try:
        time = range(len(signal))
        time = [float(x) / sampling_rate for x in time]
    except Exception as e:
        print("time", e)
        time = []

    try:
        ds = 1/sampling_rate
        energy = np.sum(signal**2*ds)
    except Exception as e:
        print("energy", e)   
        energy = []
    # end helpers
    
    # signal max
    try:
        _max = np.max(signal)
    except Exception as e:
        print("max", e)   
        _max = None
    args += [_max]
    names += ['max']
    
    # signal min
    try:
        _min = np.min(signal)
    except Exception as e:
        print("min", e)   
        _min = None
    args += [_min]
    names += ['min']
    
    # range
    try:
        _range = np.max(signal) - np.min(signal)
    except Exception as e:
        print("range", e)   
        _range = None
    args += [_range]
    names += ['range']

    # interquartile range 
    try:
        _iqr = iqr(signal)
    except Exception as e:
        print("iqr", e)
        _iqr = None
    args += [_iqr]
    names += ['iqr']

    # mean
    try:
        mean = np.mean(signal)
    except Exception as e:
        print("mean", e)   
        mean = None
    args += [mean]
    names += ['mean']

    # std
    try:
        std = np.std(signal) 
    except Exception as e:
        print(e)   
        std = None
    args += [std]
    names += ['std']

    # max to mean
    try:
        max_to_mean = np.max(signal - mean)
    except Exception as e:
        print("max_to_mean", e)   
        max_to_mean = None
    args += [max_to_mean]
    names += ['max_to_mean']

    # distance
    try:
        dist = np.sum([1 if d == 0 else d for d in np.abs(sig_diff)]) + 1
    except Exception as e:
        print("dist", e)   
        dist = None
    args += [dist]
    names += ['dist']

    # mean absolute differences
    try:
        mean_AD1 = np.mean(np.abs(sig_diff))
    except Exception as e:
        print(e)   
        mean_AD1 = None
    args += [mean_AD1]
    names += ['mean_AD1']

    # median absolute differences
    try:
        med_AD1 = np.median(np.abs(sig_diff))
    except Exception as e:
        print(e)
        med_AD1 = None
    args += [med_AD1]
    names += ['med_AD1']

    # min absolute differences
    try:
        min_AD1 = np.min(np.abs(sig_diff))
    except Exception as e:
        print(e)   
        min_AD1 = None
    args += [min_AD1]
    names += ['min_AD1']

    # max absolute differences
    try:
        max_AD1 = np.max(np.abs(sig_diff))
    except Exception as e:
        print(e)   
        max_AD1 = None
    args += [max_AD1]
    names += ['max_AD1']

    # mean of differences
    try:
        mean_D1 = np.mean(sig_diff)
    except Exception as e:
        print(e)  
        mean_D1 = None
    args += [mean_D1]
    names += ['mean_D1']

    # median of differences
    try:
        med_D1 = np.median(sig_diff)
    except Exception as e:
        print(e)  
        med_D1 = None
    args += [med_D1]
    names += ['med_D1']

    # std of differences
    try:
        std_D1 = np.std(sig_diff)
    except Exception as e:
        print(e)  
        std_D1 = None
    args += [std_D1]
    names += ['std_D1']
    
    # max of differences
    try:
        max_D1 = np.max(sig_diff)
    except Exception as e:
        print(e)  
        max_D1 = None
    args += [max_D1]
    names += ['max_D1']

    # min of differences
    try:
        min_D1 = np.min(sig_diff)
    except Exception as e:
        print(e)   
        min_D1 = None
    args += [min_D1]
    names += ['min_D1']

    # sum of differences
    try:
        sum_D1 = np.sum(sig_diff)
    except Exception as e:
        print("sum_D1", e)
        sum_D1 = None
    args += [sum_D1]
    names += ['sum_D1']

    # range of differences
    try:
        range_D1 = np.max(sig_diff) - np.min(sig_diff)
    except Exception as e:
        print("range_D1", e)
        range_D1 = None
    args += [range_D1]
    names += ['range_D1']

    # interquartile range of differences
    try:
        iqr_D1 = iqr(sig_diff)
    except Exception as e:
        print("iqr_D1", e)
        iqr_D1 = None
    args += [iqr_D1]
    names += ['iqr_D1']

    # mean of 2nd differences
    try:
        mean_D2 = np.mean(sig_diff_2)
    except Exception as e:
        print("mean_D2", e)   
        mean_D2 = None
    args += [mean_D2]
    names += ['mean_D2']

    # std of 2nd differences
    try:
        std_D2 = np.std(sig_diff_2)
    except Exception as e:
        print("std_D2", e)   
        std_D2 = None
    args += [std_D2]
    names += ['std_D2']
    
    # max of 2nd differences
    try:
        max_D2 = np.max(sig_diff_2)
    except Exception as e:
        print("max_D2", e)   
        max_D2 = None
    args += [max_D2]
    names += ['max_D2']

    # min of 2nd differences
    try:
        min_D2 = np.min(sig_diff_2)
    except Exception as e:
        print("min_D2", e)   
        min_D2 = None
    args += [min_D2]
    names += ['min_D2']

    # sum of 2nd differences
    try:
        sum_D2 = np.sum(sig_diff_2)
    except Exception as e:
        print("sum_D2", e)   
        sum_D2 = None
    args += [sum_D2]
    names += ['sum_D2']

    # range of 2nd differences
    try:
        range_D2 = np.max(sig_diff_2) - np.min(sig_diff_2)
    except Exception as e:
        print("range_D2", e)   
        range_D2 = None
    args += [range_D2]
    names += ['range_D2']

    # interquartile range of 2nd differences
    try:
        iqr_D2 = iqr(sig_diff_2)
    except Exception as e:
        print("iqr_D2", e)
        iqr_D2 = None
    args += [iqr_D2]
    names += ['iqr_D2']

    # autocorrelation sum
    try:
        if np.sum(np.abs(signal)) > 0: 
            autocorr = np.sum(np.correlate(signal, signal, 'full'))
        else:
            autocorr = None
    except Exception as e:
        print("autocorr", e)   
        autocorr = None
    args += [autocorr]
    names += ['autocorr']

    # zero_cross
    try:
        zero_cross = len(np.where(np.abs(np.diff(np.sign(signal))) >= 1)[0])
    except Exception as e:
        print("zero_cross", e)   
        zero_cross = None
    args += [zero_cross]
    names += ['zero_cross']

    # number of minimum peaks
    try:
        min_peaks = len(tools.find_extrema(signal, "min")["extrema"])
    except Exception as e:
        print("min_peaks", e)   
        min_peaks = None
    args += [min_peaks]
    names += ['min_peaks']

    # number of maximum peaks
    try:
        max_peaks = len(tools.find_extrema(signal, "max")["extrema"])
    except Exception as e:
        print("max_peaks", e)   
        max_peaks = None
    args += [max_peaks]
    names += ['max_peaks']
    
    # total energy
    try:
        total_e = np.sum(energy)
    except Exception as e:
        print("total_e", e)   
        total_e = None
    args += [total_e]
    names += ['total_e']

    _t = np.array(time).reshape(-1, 1)
    try:
        reg = linear_model.LinearRegression().fit(_t,  signal) 
        lin_reg_slope = reg.coef_[0]
    except Exception as e:
        print("lin_reg_slope", e)
        lin_reg_slope = None
    args += [lin_reg_slope]
    names += ['lin_reg_slope']

    try:
        lin_reg_b = reg.intercept_
    except Exception as e:
        print("lin_reg_b", e)   
        lin_reg_b = None
    args += [lin_reg_b]
    names += ['lin_reg_b']

    try:
        c = 0
        if np.sum(np.abs(signal)) > 0 and len(np.unique(signal)) > 1:
            _r = reg.predict(_t)
            if np.sum(np.abs(_r)) > 0 and len(np.unique(_r)) > 1:
                corr_lin_reg = pearson_correlation(signal, reg.predict(_t))[0]
                c = 1
        if not c:
            corr_lin_reg = None
    except Exception as e:
        print("corr_lin_reg", e)  
        corr_lin_reg = None
    args += [corr_lin_reg]
    names += ['corr_lin_reg']

    # hjorth features
    # mobility
    try:    
        mobility = hjorth_mob(signal)['mobility']
    except Exception as e:
        print("mobility", e)  
        mobility = None
    args += [mobility]
    names += ['mobility']

    # complexity
    try:    
        complexity = hjorth_comp(signal)['complexity']
    except Exception as e:
        print("complexity", e)  
        complexity = None
    args += [complexity]
    names += ['complexity']

    # chaos
    try:    
        _chaos = hjorth_chaos(signal)['chaos']
    except Exception as e:
        print("chaos", e)  
        _chaos = None
    args += [_chaos]
    names += ['chaos']

    # Hazard
    try:    
        if hjorth_chaos(signal)['chaos'] is not None:
            hazard = hjorth_chaos(sig_diff)['chaos'] / hjorth_chaos(signal)['chaos']
        else:
            hazard = None
    except Exception as e:
        print("hazard", e)  
        hazard = None
    args += [hazard]
    names += ['hazard']

    # kurtosis
    try:
        kurtosis = stats.kurtosis(signal, bias=False)
    except Exception as e:
        print("kurtosis", e) 
        kurtosis = None
    args += [kurtosis]
    names += ['kurtosis']

    # skewness
    try:
        skewness = stats.skew(signal, bias=False)
    except Exception as e:
        print("skewness", e) 
        skewness = None
    args += [skewness]
    names += ['skewness']

    # root mean square
    try:
        rms = np.sqrt(np.sum(signal ** 2) / len(signal))
    except Exception as e:
        print("rms", e) 
        rms = None
    args += [rms]
    names += ['rms']

    # midhinge
    try:
        quant = np.quantile(signal, [0.25, 0.5, 0.75])
        midhinge = (quant[0] + quant[2])/2
    except Exception as e:
        print("midhinge", e) 
        midhinge = None
    args += [midhinge]
    names += ['midhinge']

    # trimean
    try:
        trimean = (quant[1] + midhinge)/2
    except Exception as e:
        print("trimean", e) 
        trimean = None
    args += [trimean]
    names += ['trimean']

    # histogram
    try:
        _hist = list(np.histogram(signal, bins=5)[0])
        _hist = _hist/np.sum(_hist)
    except Exception as e:
        print("time hist", e) 
        _hist = [None] * 5
    args += [i for i in _hist]
    names += ['stat_hist' + str(i) for i in range(len(_hist))]

    # entropy
    try:
        if np.sum(np.abs(signal)) > 0: 
            _entropy = np.nan_to_num(entropy(signal))
        else:
            _entropy = None
    except Exception as e:
        print("entropy", e) 
        _entropy = None
    args += [_entropy]
    names += ['entropy']

    # output
    args = tuple(args)
    names = tuple(names)

    return utils.ReturnTuple(args, names)
