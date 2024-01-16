# -*- coding: utf-8 -*-
"""
biosppy.quality
-------------

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


def quality_eda(x=None, methods=['bottcher'], sampling_rate=None):
    """Compute the quality index for one EDA segment.

        Parameters
        ----------
        x : array
            Input signal to test.
        methods : list
            Method to assess quality. One or more of the following: 'bottcher'.
        sampling_rate : int
            Sampling frequency (Hz).
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
            quality = eda_sqi_bottcher(x, sampling_rate)
    
        args += (quality,)
        names += (method,)

    return utils.ReturnTuple(args, names)


def quality_ecg(segment, methods=['3Level'], sampling_rate=None, 
                fisher=True, f_thr=0.01, threshold=0.9, bit=0, 
                nseg=1024, num_spectrum=[], dem_spectrum=[], 
                mode_fsqi='simple'):
    
    """Compute the quality index for one ECG segment.

    Parameters
    ----------
    segment : array
        Input signal to test.
    method : string
        Method to assess quality. One of the following: '3Level', 'pSQI', 'kSQI', 'Zhao'.
    sampling_rate : int
        Sampling frequency (Hz).
    threshold : float
        Threshold for the correlation coefficient.
    bit : int
        Number of bits of the ADC.? Resolution bits, for the BITalino is 10 bits.

    Returns
    -------
    args : tuple
        Tuple containing the quality index for each method.
    names : tuple
        Tuple containing the name of each method.
    """
    args, names = (), ()

    for method in methods:

        assert method in ['Level3', 'pSQI', 'kSQI', 'Zhao'], 'Method should be one of the following: 3Level, pSQI, kSQI, Zhao'

        if method == 'Level3':
            # returns a SQI level 0, 0.5 or 1.0
            quality = ecg_sqi_level3(segment, sampling_rate, threshold, bit)

        elif method == 'pSQI':
            quality = ecg.pSQI(segment, f_thr=f_thr)
        
        elif method == 'kSQI':
            quality = ecg.kSQI(segment, fisher=fisher)

        elif method == 'fSQI':
            quality = ecg.fSQI(segment, fs=sampling_rate, nseg=nseg, num_spectrum=num_spectrum, dem_spectrum=dem_spectrum, mode=mode_fsqi)

        args += (quality,)
        names += (method,)

    return utils.ReturnTuple(args, names)


def quality_ppg(x=None, methods=['glasstetter'], sampling_rate=None, q_thr=0.8):
    """Compute the quality index for one PPG segment.

    Parameters
    ----------
    x : array
        Input signal to test.
    methods : list
        Methods to assess quality. One or more of the following: 'glasstetter'.
    sampling_rate : int
        Sampling frequency (Hz).
    q_thr : float
        Threshold for the spectral entropy acceptability.

    Returns
    -------
    args : tuple
        Tuple containing the quality index for each method.
    names : tuple
        Tuple containing the name of each method.
    """

    args, names = (), ()
    available_methods = ['glasstetter']

    for method in methods:

        assert method in available_methods, 'Method should be one or more of the following: ' + ', '.join(available_methods)

        if method == 'glasstetter':
            quality = ppg_sqi(x, sampling_rate, q_thr=q_thr)
        
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


def eda_sqi_bottcher(x=None, sampling_rate=None):  # -> Timeline
    """
    Suggested by Böttcher et al. Scientific Reports, 2022, for wearable wrist EDA.

    This is given by a binary score 0/1 defined by the following rules:
     - mean of the segment of 2 seconds should be > 0.05
     - rate of amplitude change (given by racSQI) should be < 0.2
    This score is calculated for each 2 seconds window of the segment. The average of the scores is the final SQI.

    This method was designed for a segment of 60s

    """
    quality_score = 0
    if x is None:
        raise TypeError("Please specify the input signal.")
    if sampling_rate is None:
        raise TypeError("Please specify the sampling rate.")
    
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
    

def spectral_entropy(x, sampling_rate, nperseg, fmin, fmax):
    """Spectral entropy for a PPG Signal.

    As proposed by Glasstetter et al. MDPI Sensors, 21, 2021.
    "Identification of Ictal Tachycardia in Focal Motor- and Non-Motor Seizures by Means of a Wearable PPG Sensor."

    Parameters
    ----------
    x : array
        Input signal.
    sampling_rate : int
        Sampling frequency (Hz).
    nperseg : int
        Length of each segment.
    fmin : float
        Minimum frequency (Hz).
    fmax : float
        Maximum frequency (Hz).
    
    Returns
    -------
    entropy_norm : float
        Normalized entropy.
    """

    assert len(x) >= nperseg, 'Segment must be 4s long'
    
    # if nperseg = 4s, then 3.75 s of overlap
    noverlap = int(0.9375 * nperseg)  
    
    # use the welch spectrum to compute the PSD and the frequency vector
    f, psd = tools.welch_spectrum(signal=x, sampling_rate=sampling_rate, size=nperseg, overlap=noverlap)
    
    # select the frequency band of interest
    idx_min = np.argmin(np.abs(f - fmin))
    idx_max = np.argmin(np.abs(f - fmax))
    psd = psd[idx_min:idx_max]
    # normalize the PSD
    psd /= np.sum(psd)  
    entropy = - np.sum(psd * np.log2(psd))
    N = idx_max - idx_min
    entropy_norm = entropy / np.log2(N)
    return entropy_norm


def ppg_sqi(x, sampling_rate, q_thr=0.8):
    """PPG SQI as proposed by Glasstetter et al. MDPI Sensors, 21, 2021.
    "Identification of Ictal Tachycardia in Focal Motor- and Non-Motor Seizures by Means of a Wearable PPG Sensor."
    Also used in Böttcher et al. Scientific Reports, 2022.
    "Data Quality Monitoring
    
    Parameters
    ----------
    x : array
        Input signal.
    sampling_rate : int
        Sampling frequency (Hz).
    q_thr : float
        Quality threshold. Used as 0.72 in Glasstetter et al. and 0.8 in Böttcher et al. The higher, more lower quality segments are accepted.
    
    Returns
    -------
    quality_score : float
        Quality score ranging between 0 and 1. Average of the binary quality scores of each 4s segment with 3.75 overlap.
    """

    nperseg = int(4 * sampling_rate)  # 4 s window
    fmin = 0.1  # Hz
    fmax = 5  # Hz

    sp_ent = [spectral_entropy(xi, sampling_rate, nperseg, fmin, fmax) for xi in x.reshape(-1, nperseg)]

    quality_score = np.mean((sp_ent < q_thr).astype(int))

    return quality_score