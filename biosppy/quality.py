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


def quality_ecg(segment, methods=['Level3'], sampling_rate=None, 
                fisher=True, f_thr=0.01, threshold=0.9, bit=0, 
                nseg=1024, num_spectrum=[5, 20], dem_spectrum=None, 
                mode_fsqi='simple'):
    
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

    Returns
    -------
    args : tuple
        Tuple containing the quality index for each method.
    names : tuple
        Tuple containing the name of each method.
    """
    args, names = (), ()
    available_methods = ['Level3', 'pSQI', 'kSQI', 'fSQI']

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
    
