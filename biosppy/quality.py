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
from scipy.signal import resample

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


def quality_resp(segment, methods=['charlton'], sampling_rate=None):
    """
    Compute the quality index for one Respiration segment.
    
    Parameters
    ----------
    segment : array
        Input signal to test.
    methods : list
        Method to assess quality. One or more of the following: ''.
    sampling_rate : int
        Sampling frequency (Hz).
    
    Returns
    -------
    args : tuple
        Tuple containing the quality index for each method.
    names : tuple
        Tuple containing the name of each method.
    """
    args, names = (), ()
    available_methods = ['charlton']

    for method in methods:

        assert method in available_methods, 'Method should be one of the following: ' + ', '.join(available_methods)

        if method == 'charlton':
            quality = resp_sqi(segment, sampling_rate)

        args += (quality,)
        names += (method,)

    return utils.ReturnTuple(args, names)


def resp_sqi(segment, sampling_rate):

    """
    Calculate the SQI of the respiratory signal following the approach of:
    (Charlton et al. 2021) "An impedance pneumography signal quality index: 
    Design, assessment and application to respiratory rate monitoring"
    ---
    Parameters:
        segment: np.array. The segment of the signal to be analyzed
        sampling_rate: int. The sampling rate of the signal
    ---
    Returns:
        sqi: float. The SQI of the signal
    """
    
    if sampling_rate is None:
        raise IOError('Sampling frequency is required')
    if sum(segment) < 1:
        return 0
    
    if len(segment) != sampling_rate * 32:
        raise IOError('Segment must be 32s long')
    
    # BREATH DETECTION
    # divide into 32s segments
    # 1) Low pass filtered above 1Hz and downsampled to 5Hz
    segment_filt = tools.filter_signal(segment, ftype='butter', band='bandpass', frequency=[0.1, 1.], order=2,
                                sampling_rate=sampling_rate)['signal']
    # downsampling to 5 Hz
    segment_filt_downsampled = resample(segment_filt, int(len(segment_filt)*5 / sampling_rate))

    # 2) Normalisation (mean=0, std=1)
    norm_sig = (segment_filt_downsampled - np.mean(segment_filt_downsampled)) / np.std(segment_filt_downsampled)
    # 3) Count Orig algorithm

    # 1) Identify peaks and troughs as local extremas
    peaks = tools.find_extrema(norm_sig, mode='max')['extrema']
    troughs = tools.find_extrema(norm_sig, mode='min')['extrema']
    peaks_amplitudes = norm_sig[peaks]
    troughs_amplitudes = norm_sig[troughs]

    # 2) relevant peaks identified
    peaks_rel = np.array(peaks)[peaks_amplitudes > 0.2 * np.percentile(peaks_amplitudes, 75)]
    # 3) relevant troughs identified
    troughs_rel = np.array(troughs)[troughs_amplitudes < 0.2 * np.percentile(troughs_amplitudes, 25)]

    # 4) 1 peak per consecutive troughs
    try:
        peaks_idx = np.hstack([np.where(((peaks_rel > troughs_rel[i]) * (peaks_rel < troughs_rel[i + 1])))[0] for i in
                            range(len(troughs_rel) - 1)])
    except:
        return 0
    # find peaks rel values of the indexes identified in peaks_idx
    peaks_val = peaks_rel[peaks_idx]
    # breaths will be the times between consecutive peaks found between troughs
    time_breaths = np.diff(peaks_val)
    if len(time_breaths) < 1:
        return 0

    # Evaluate valid breaths
    # 1) std time breaths > 0.25
    quality = True
    if np.std(time_breaths) < 0.25:
        quality = False
        return 1 if quality else 0 # there is signal but with low quality
    # 2) 15% of time breaths > 1.5 or < 0.5 * median breath duration
    bad_breaths = time_breaths[
        ((time_breaths > (1.5 * np.median(time_breaths))) & (time_breaths < (0.5 * np.median(time_breaths))))]
    ratio = len(bad_breaths) / len(time_breaths)
    if ratio >= 0.15:
        quality = False
        return 1 if quality else 0.1 # there is signal but with low quality
    # 3) 60% of segment is occupied by valid breaths
    if (sum(time_breaths) / len(norm_sig)) < 0.6:
        quality = False
        return 1 if quality else 0 # there is signal but with low quality

    # assess similarity of breath morphologies
    # calculate template breath
    # calculate correlation between individual breaths and the template
    # 4) is the mean correlation coeffient > 0.75?
    # get mean breath interval
    mean_breath_interval = np.mean(time_breaths)
    breaths = [norm_sig[int(peaks_val[i] - mean_breath_interval // 2): int(peaks_val[i] + mean_breath_interval // 2)]
            for i in range(len(peaks_val)) if ((peaks_val[i] - mean_breath_interval//2) >= 0)
            and (peaks_val[i] + mean_breath_interval//2 <= len(norm_sig))]

    mean_template = np.mean(breaths, axis=0)


    # compute correlation and mean correlation coefficient
    mean_corr = np.mean([stats.pearsonr(breath, mean_template)[0] for breath in breaths])
    # check if mean corr > 0.75
    if mean_corr < 0.75:
        quality = False

    return 1 if quality else 0 # there is signal but with low quality




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
        print('cSQI Advice (comment this by setting verbose=0) -> The original segment should be more than 30s long for optimal results.')

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
        print('hosSQI Advice (comment this by setting verbose=0) -> The signal should be filtered before this SQI and 5s long.')
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
    