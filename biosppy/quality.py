# -*- coding: utf-8 -*-
"""
biosppy.quality
----------------

This provides functions to assess the quality of several biosignals.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function

# local
from . import utils
from .signals import ecg, tools

# 3rd party
import numpy as np
from scipy import stats

def quality_eda(x=None, methods=['bottcher'], sampling_rate=None, verbose=1):
    """Compute the quality index for one EDA segment.

        Parameters
        ----------
        x : array
            Input signal to test.
        methods : list
            Method to assess quality. One or more of the following: 'bottcher'.
        sampling_rate : int
            Sampling frequency (Hz).
        verbose : int
            If 1, a commentary is printed regarding the quality of the signal and details of the function. Default is 1.

        Returns
        -------
        args : tuple
            Tuple containing the quality index for each method.
        names : tuple
            Tuple containing the name of each method.
        """
    # check inputs
    if x is None:
        raise TypeError("Please specify the input signal.")
    
    if sampling_rate is None:
        raise TypeError("Please specify the sampling rate.")
    
    assert len(x) > sampling_rate * 2, 'Segment must be 5s long'

    args, names = (), ()
    available_methods = ['bottcher']

    for method in methods:

        assert method in available_methods, "Method should be one of the following: " + ", ".join(available_methods)
    
        if method == 'bottcher':
            quality = eda_sqi_bottcher(x, sampling_rate, verbose)
    
        args += (quality,)
        names += (method,)

    return utils.ReturnTuple(args, names)


def quality_ecg(segment, methods=['Level3'], sampling_rate=None, 
                fisher=True, f_thr=0.01, threshold=0.9, bit=0, 
                nseg=1024, num_spectrum=[5, 20], dem_spectrum=None, 
                mode_fsqi='simple', verbose=1):
    
    """Compute the quality index for one ECG segment.

    Parameters
    ----------
    segment : array
        Input signal to test.
    method : string
        Method to assess quality. One of the following: 'Level3', 'pSQI', 'kSQI', 'fSQI'.
    sampling_rate : int
        Sampling frequency (Hz).
    threshold : float
        Threshold for the correlation coefficient.
    bit : int
        Number of bits of the ADC. Resolution bits, for the BITalino is 10 bits.
    verbose : int
        If 1, a commentary is printed regarding the quality of the signal and details of the function. Default is 1.

    Returns
    -------
    args : tuple
        Tuple containing the quality index for each method.
    names : tuple
        Tuple containing the name of each method.
    """
    args, names = (), ()
    available_methods = ['Level3', 'pSQI', 'kSQI', 'fSQI', 'cSQI', 'hosSQI']

    for method in methods:

        assert method in available_methods, 'Method should be one of the following: ' + ', '.join(available_methods)

        if method == 'Level3':
            # returns a SQI level 0, 0.5 or 1.0
            quality = ecg_sqi_level3(segment, sampling_rate, threshold, bit)

        elif method == 'pSQI':
            quality = ecg.pSQI(segment, f_thr=f_thr)
        
        elif method == 'kSQI':
            quality = ecg.kSQI(segment, fisher=fisher)

        elif method == 'fSQI':
            quality = ecg.fSQI(segment, fs=sampling_rate, nseg=nseg, num_spectrum=num_spectrum, dem_spectrum=dem_spectrum, mode=mode_fsqi)
        
        elif method == 'cSQI':
            rpeaks = ecg.hamilton_segmenter(segment, sampling_rate=sampling_rate)['rpeaks']
            quality = cSQI(rpeaks, verbose)
        
        elif method == 'hosSQI':
            quality = hosSQI(segment, verbose)

        args += (quality,)
        names += (method,)

    return utils.ReturnTuple(args, names)


def ecg_sqi_level3(segment, sampling_rate, threshold, bit):

    """Compute the quality index for one ECG segment. The segment should have 10 seconds.


    Parameters
    ----------
    segment : array
        Input signal to test.
    sampling_rate : int
        Sampling frequency (Hz).
    threshold : float
        Threshold for the correlation coefficient.
    bit : int
        Number of bits of the ADC.? Resolution bits, for the BITalino is 10 bits.
    
    Returns
    -------
    quality : string
        Signal Quality Index ranging between 0 (LQ), 0.5 (MQ) and 1.0 (HQ).

    """
    LQ, MQ, HQ = 0.0, 0.5, 1.0
    
    if bit !=  0:
        if (max(segment) - min(segment)) >= (2**bit - 1):
            return LQ
    if sampling_rate is None:
        raise IOError('Sampling frequency is required')
    if len(segment) < sampling_rate * 5:
        raise IOError('Segment must be 5s long')
    else:
        # TODO: compute ecg quality when in contact with the body
        rpeak1 = ecg.hamilton_segmenter(segment, sampling_rate=sampling_rate)['rpeaks']
        rpeak1 = ecg.correct_rpeaks(signal=segment, rpeaks=rpeak1, sampling_rate=sampling_rate, tol=0.05)['rpeaks']
        if len(rpeak1) < 2:
            return LQ
        else:
            hr = sampling_rate * (60/np.diff(rpeak1))
            quality = MQ if (max(hr) <= 200 and min(hr) >= 40) else LQ
        if quality == MQ:
            templates, _ = ecg.extract_heartbeats(signal=segment, rpeaks=rpeak1, sampling_rate=sampling_rate, before=0.2, after=0.4)
            corr_points = np.corrcoef(templates)
            if np.mean(corr_points) > threshold:
                quality = HQ

    return quality 


def eda_sqi_bottcher(x=None, sampling_rate=None, verbose=1):  # -> Timeline
    """ Suggested by Böttcher et al. Scientific Reports, 2022, for wearable wrist EDA.
    This is given by a binary score 0/1 defined by the following rules:
    - mean of the segment of 2 seconds should be > 0.05
    - rate of amplitude change (given by racSQI) should be < 0.2
    This score is calculated for each 2 seconds window of the segment. The average of the scores is the final SQI.
    This method was designed for a segment of 60s

    Parameters
    ----------
    x : array
        Input signal to test.
    sampling_rate : int
        Sampling frequency (Hz).
    verbose : int
        If 1, a commentary is printed regarding the quality of the signal and details of the function. Default is 1.
    
    Returns
    -------
    quality_score : string
        Signal Quality Index.
    """
    quality_score = 0
    if x is None:
        raise TypeError("Please specify the input signal.")
    if sampling_rate is None:
        raise TypeError("Please specify the sampling rate.")
    if verbose == 1:
        if len(x) < sampling_rate * 60:
            print("This method was designed for a signal of 60s but will be applied to a signal of {}s".format(len(x)/sampling_rate))
    # create segments of 2 seconds
    segments_2s = x.reshape(-1, int(sampling_rate*2))
    ## compute racSQI for each segment
    # first compute the min and max of each segment
    min_ = np.min(segments_2s, axis=1)
    max_ = np.max(segments_2s, axis=1)
    # then compute the RAC (max-min)/max
    rac = np.abs((max_ - min_) / max_)
    # ratio will be 1 if the rac is < 0.2 and if the mean of the segment is > 0.05 and will be 0 otherwise
    quality_score = ((rac < 0.2) & (np.mean(segments_2s, axis=1) > 0.05)).astype(int)
    # the final SQI is the average of the scores 
    return np.mean(quality_score)
    

def cSQI(rpeaks=None, verbose=1):
    """For the ECG signal
    Calculate the Coefficient of Variation of RR Intervals (cSQI).
    Parameters
    ----------
    rpeaks : array-like
        Array containing R-peak locations. Should be filtered? How many seconds are adequate?
    verbose : int
        If 1, a commentary is printed regarding the quality of the signal and details of the function. Default is 1.
    Returns
    -------
    cSQI : float
        Coefficient of Variation of RR Intervals. cSQI - best near 0
    References
    ----------
    ..  [Zhao18] Zhao, Z., & Zhang, Y. (2018).
    SQI quality evaluation mechanism of single-lead ECG signal based on simple heuristic fusion and fuzzy comprehensive evaluation.
    Frontiers in Physiology, 9, 727.
    """
    if rpeaks is None:
        raise TypeError("Please specify the R-peak locations.")
  
    rr_intervals = np.diff(rpeaks)
    sdrr = np.std(rr_intervals)
    mean_rr = np.mean(rr_intervals)
    cSQI = sdrr / mean_rr

    if verbose == 1:
        print('-------------------------------------------------------') 
        print('cSQI Advice (remove this by setting verbose=0) -> The original segment should be more than 30s long for optimal results.')

        if cSQI < 0.45:
            str_level = "Optimal"
        elif 0.45 <= cSQI <= 0.64:
            str_level = "Suspicious"
        else:
            str_level = "Unqualified"

        print('cSQI is {:.2f} -> {str_level}'.format(cSQI, str_level= str_level))
        print('-------------------------------------------------------') 
    
    return cSQI
    

def hosSQI(signal=None, quantitative=False, verbose=1):
    """For the ECG signal.
    Calculate the Higher-order-statistics-SQI (hosSQI).
    Parameters
    ----------
    signal : array-like
        ECG signal. Should be filtered? How many seconds are adequate?

    verbose : bool
        If True, a warning message is printed. Default is True.
    Returns
    -------
    hosSQI : float
        Higher-order-statistics-SQI. hosSQI - best near 1
    References
    ----------
    .. [Nardelli20] Nardelli, M., Lanata, A., Valenza, G., Felici, M., Baragli, P., & Scilingo, E.P. (2020).
    A tool for the real-time evaluation of ECG signal quality and activity: Application to submaximal treadmill test in horses.
    Biomedical Signal Processing and Control, 56, 101666. doi: 10.1016/j.bspc.2019.101666.
    .. [Rahman22] Rahman, Md. Saifur, Karmakar, Chandan, Natgunanathan, Iynkaran, Yearwood, John, & Palaniswami, Marimuthu. (2022).
    Robustness of electrocardiogram signal quality indices.
    Journal of The Royal Society Interface, 19. doi: 10.1098/rsif.2022.0012.
    """
    # signal should be filtered?
    if signal is None:
        raise TypeError("Please specify the ECG signal.")

    kSQI = stats.kurtosis(signal)
    sSQI = stats.skew(signal)
    print('kurtosis: ', kSQI)
    print('skewness: ', sSQI)

    hosSQI = abs(sSQI) * kSQI / 5

    if verbose == 1:
        print('-------------------------------------------------------') 
        print('hosSQI Advice (remove this by setting verbose=0) -> The signal must be at least 5s long and should be filtered before applying this function.')
        print('hosSQI is a measure without an upper limit.')
        if hosSQI > 0.8:
            str_level = "Optimal"
        elif 0.5 < hosSQI <= 0.8:
            str_level = "Acceptable"
        else:
            str_level = "Unacceptable"
        print('hosSQI is {:.2f} -> {str_level}'.format(hosSQI, str_level= str_level))
        print('-------------------------------------------------------') 
    
    return hosSQI
