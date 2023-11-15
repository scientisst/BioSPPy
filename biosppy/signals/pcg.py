# -*- coding: utf-8 -*-
"""
biosppy.signals.pcg
-------------------

This module provides methods to process Phonocardiography (PCG) signals.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function

# 3rd party
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# local
from . import tools as st
from . import ecg
from .. import plotting, utils


def pcg(signal=None, sampling_rate=1000., units=None, path=None, show=True):
    """

    Parameters
    ----------
    signal : array
        Raw PCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    units : str, optional
        The units of the input signal. If specified, the plot will have the
        y-axis labeled with the corresponding units.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered PCG signal.
    peaks : array
        Peak location indices.
    hs: array
        Classification of peaks as S1 or S2.
    heart_rate : double
        Average heart rate (bpm).
    systolic_time_interval : double
        Average systolic time interval (seconds).
    heart_rate_ts : array
         Heart rate time axis reference (seconds).
    inst_heart_rate : array
        Instantaneous heart rate (bpm).

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # Filter Design
    order = 2
    passBand = np.array([25, 400])
    
    # Band-Pass filtering of the PCG:        
    filtered, fs, params = st.filter_signal(signal, 'butter', 'bandpass', order, passBand, sampling_rate)

    # find peaks
    peaks, envelope = find_peaks(signal=filtered, sampling_rate=sampling_rate)
    
    # classify heart sounds
    hs, = identify_heart_sounds(beats=peaks, sampling_rate=sampling_rate)
    s1_peaks = peaks[np.where(hs==1)[0]]
    
    # get heart rate
    heartRate,systolicTimeInterval = get_avg_heart_rate(envelope,sampling_rate)
    
    # get instantaneous heart rate
    hr_idx,hr = st.get_heart_rate(s1_peaks, sampling_rate)
    
    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_hr = ts[hr_idx]

    # plot
    if show:
        plotting.plot_pcg(ts=ts,
                raw=signal,
                filtered=filtered,
                peaks=peaks,
                heart_sounds=hs,
                heart_rate_ts=ts_hr,
                inst_heart_rate=hr,
                units=units,
                path=path,
                show=True)
        
        
    # output
    args = (ts, filtered, peaks, hs, heartRate, systolicTimeInterval, ts_hr, hr)
    names = ('ts', 'filtered', 'peaks', 'heart_sounds',
             'heart_rate', 'systolic_time_interval','heart_rate_ts','inst_heart_rate')

    return utils.ReturnTuple(args, names)

def find_peaks(signal=None,sampling_rate=1000.):
    
    """Finds the peaks of the heart sounds from the homomorphic envelope

    Parameters
    ----------
    signal : array
        Input filtered PCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    peaks : array
        peak location indices.
    envelope : array
        Homomorphic envelope (normalized).

    """
    
    # Compute homomorphic envelope
    envelope, = homomorphic_filter(signal,sampling_rate)
    envelope, = st.normalize(envelope)
    
    # Find the prominent peaks of the envelope
    peaksIndices, _ = ss.find_peaks(envelope, distance = 0.10*sampling_rate, prominence = 0.25)
    
    peaks = np.array(peaksIndices, dtype='int')

    return utils.ReturnTuple((peaks,envelope), 
                             ('peaks','homomorphic_envelope'))


def homomorphic_filter(signal=None, sampling_rate=1000., f_LPF=8, order=2):
    """Finds the homomorphic envelope of a signal.

    Adapted to Python from original MATLAB code written by David Springer, 2016 (C), for
    comparison purposes in the paper [Springer15]_.
    Available at: https://github.com/davidspringer/Springer-Segmentation-Code

    Follows the approach described by Schmidt et al. [Schimdt10]_.

    Parameters
    ----------
    signal : array
        Input filtered PCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    f_LPF: int, float, optional
        Low pass cut-off frequency (Hz)
    order: int, optional
        Order of Butterworth low pass filter.

    Returns
    -------
    envelope : array
        Homomorphic envelope (non-normalized).

    References
    ----------
    .. [Springer15] D.Springer, "Logistic Regression-HSMM-based Heart Sound Segmentation",
       IEEE Trans. Biomed. Eng., In Press, 2015.
    .. [Schimdt10] S. E. Schmidt et al., "Segmentation of heart sound recordings by a 
       duration-dependent hidden Markov model", Physiol. Meas., 2010

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    sampling_rate = float(sampling_rate)
    f_LPF = float(f_LPF)
    
    # Filter Design
    passBand = np.array([25, 400])
    
    # Band-Pass filtering of the PCG:        
    signal, fs, params = st.filter_signal(signal, 'butter', 'bandpass', order, passBand, sampling_rate)
    
    # LP-filter Design (to reject the oscillating component of the signal):
    b, a = ss.butter(order, 2 * f_LPF / fs, 'low')
    envelope = np.exp(ss.filtfilt(b, a, np.log(np.abs(ss.hilbert(signal)))))
    
    # Remove spurious spikes in first sample:
    envelope[0] = envelope[1]   

    return utils.ReturnTuple((envelope,), 
                             ('homomorphic_envelope',))

def get_avg_heart_rate(envelope=None, sampling_rate=1000.):
    
    """Compute average heart rate from the signal's homomorphic envelope.
    
    Follows the approach described by Schmidt et al. [Schimdt10]_, with
    code adapted from David Springer [Springer16]_.
    
    Parameters
    ----------
    envelope : array
        Signal's homomorphic envelope
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
        
    Returns
    -------
    heart_rate : double
        Average heart rate (bpm).
    systolic_time_interval : double
        Average systolic time interval (seconds).

    Notes
    -----
    * Assumes normal human heart rate to be between 40 and 200 bpm.
    * Assumes normal human systole time interval to be between 0.2 seconds and half a heartbeat
    
    References
    ----------
    .. [Schimdt10] S. E. Schmidt et al., "Segmentation of heart sound recordings by a 
       duration-dependent hidden Markov model", Physiol. Meas., 2010
    .. [Springer16] D.Springer, "Heart sound segmentation code based on duration-dependant
       HMM", 2016. Available at: https://github.com/davidspringer/Springer-Segmentation-Code   
    
    """

    # check inputs
    if envelope is None:
        raise TypeError("Please specify the signal's homomorphic envelope.")
    
    autocorrelation = np.correlate(envelope,envelope,mode='full')
    autocorrelation = autocorrelation[(autocorrelation.size)//2:]
    
    min_index = int(0.3*sampling_rate)
    max_index = int(1.5*sampling_rate)

    index = np.argmax(autocorrelation[min_index-1:max_index-1])
    true_index = index+min_index-1
    heartRate = 60/(true_index/sampling_rate)
    
    max_sys_duration = int(np.round(((60/heartRate)*sampling_rate)/2))
    min_sys_duration = int(np.round(0.2*sampling_rate))
    
    pos = np.argmax(autocorrelation[min_sys_duration-1:max_sys_duration-1])
    systolicTimeInterval = (min_sys_duration+pos)/sampling_rate
    

    return utils.ReturnTuple((heartRate,systolicTimeInterval),
                             ('heart_rate','systolic_time_interval'))

def identify_heart_sounds(beats = None, sampling_rate = 1000.):
    
    """Classify heart sound peaks as S1 or S2
     
    Parameters
    ----------
    beats : array
        Peaks of heart sounds
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
        
    Returns
    -------
    classification : array
        Classification of heart sound peaks. 1 is S1, 2 is S2
    
    """

    one_peak_ahead = np.roll(beats, -1)

    SS_intervals = (one_peak_ahead[0:-1] - beats[0:-1]) / sampling_rate
    
    # Initialize the vector to store the classification of the peaks:
    classification = np.zeros(len(beats))
        
    # Classify the peaks. 
    # Terrible algorithm, but good enough for now    
    for i in range(1,len(beats)-1):
        if SS_intervals[i-1] > SS_intervals[i]:
            classification[i] = 0
        else:
            classification[i] = 1
    classification[0] = int(not(classification[1]))
    classification[-1] = int(not(classification[-2]))    
    
    classification += 1    
        
    return utils.ReturnTuple((classification,), ('heart_sounds',))

def ecg_based_segmentation(pcg_signal=None, ecg_signal=None, sampling_rate=1000.0, show=False):
    """Assign state labels to PCG recording based on markers from simultaneous ECG signal.

    Adapted to Python from original MATLAB code written by David Springer, 2016 (C), for 
    comparison purposes in the paper [Springer15]_.
    Available at: https://github.com/davidspringer/Springer-Segmentation-Code 
    
    Heart sounds timing durations were obtained from [Schimdt10]_.
    
    Parameters
    ----------
    pcg_signal : array
        PCG signal to be segmented.
    ecg_signal : array
        Simultaneous ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz) of both signals.
    show : bool, optional
        If True, show a plot with the segmented signal.
        
    Returns
    -------
    states : array
        State labels of PCG recording

    References
    ----------
    .. [Springer15] D.Springer, "Logistic Regression-HSMM-based Heart Sound Segmentation",
       IEEE Trans. Biomed. Eng., In Press, 2015.
    .. [Schimdt10] S. E. Schmidt et al., "Segmentation of heart sound recordings by a 
       duration-dependent hidden Markov model", Physiol. Meas., 2010    
    """
    
    # ensure numpy
    pcg_signal = np.array(pcg_signal)
    ecg_signal = np.array(ecg_signal)
    sampling_rate = float(sampling_rate)    

    # Compute homomorphic envelope to find peaks
    envelope, = homomorphic_filter(pcg_signal, sampling_rate=sampling_rate)
    envelope, = st.normalize(envelope)
        
    states = np.zeros((len(envelope),))
    
    # Extract locations for R peaks and end T-waves
    ecg_out = ecg.ecg(ecg_signal, sampling_rate=sampling_rate, show=False)
    rpeaks = ecg_out['rpeaks'].astype('int64')
    t_positions = ecg.getTPositions(ecg_out)
    t_ends = t_positions["T_end_positions"]
    
    #Timing durations from Schmidt:
    mean_S1 = 0.122*sampling_rate
    std_S1 = 0.022*sampling_rate
    mean_S2 = 0.092*sampling_rate
    std_S2 = 0.022*sampling_rate

    # Set duration from each R-peak to (R-peak+mean_S1) as first state S1.
    # Upper and lower bounds are set to avoid errors of searching outside size of the signal
    for peak in rpeaks:

        upper_bound = int(min([len(states), peak + mean_S1]))
        states[peak:min([upper_bound-1,len(states)])] = 1
    
    # Set S2 as state 3 depending on position of end T-wave in ECG
    for t_end in t_ends:
        
        # find search window of envelope: T-end +- mean+1std
        lower_bound = int(max([t_end - np.floor((mean_S2 + std_S2)),1]))
        upper_bound = int(min([len(states), np.ceil(t_end + np.floor(mean_S2 + std_S2))]))

        search_window = envelope[lower_bound-1:upper_bound]*(states[lower_bound-1:upper_bound]!=1).ravel()
        
        # Find the maximum value of the envelope in the search window:
        S2_index = np.argmax(search_window)
        
        # Find the actual index in the envelope of the maximum peak.
        # Make sure this has a max value of the length of the signal:
        S2_index = min([len(states),lower_bound+ S2_index-1])
        
        # Set the states to state 3, centered on the S2 peak, +- 1/2 of the
        # expected S2 sound duration.
        upper_bound = int(min([len(states), np.ceil(S2_index +((mean_S2)/2))]))
        states[int(max([np.ceil(S2_index - ((mean_S2)/2)),1]))-1:upper_bound] = 3
        
        # Set the spaces between state 3 and the next R peak as state 4:
        # Find the next rpeak after this S2. Exclude those that happened before by setting them to infinity.
        diffs = (rpeaks - t_end).astype(float)
        diffs[diffs<0] = np.inf
        
        # If the array is empty, then no S1s after this S2, so set to end of signal:
        if np.size(diffs[diffs<np.inf])==0:
            end_pos = len(states)
        else:
            # else, send the end position to the minimum diff
            index = np.argmin(diffs)
            end_pos = rpeaks[index]
            
        states[int(np.ceil(S2_index +((mean_S2 +(0*std_S2))/2))-1):end_pos] = 4
    
    # Set first and last sections of the signal (before first R-peak, and after last end T-wave)
    first_location_of_definite_state = np.argwhere(states!=0)[0][0]

    if first_location_of_definite_state > 0:
        if states[first_location_of_definite_state] == 1:
            states[0:first_location_of_definite_state] = 4
        
        if states[first_location_of_definite_state] == 3:
            states[0:first_location_of_definite_state+1] = 2    
    
    last_location_of_definite_state = np.argwhere(states!=0)[-1][0]
    
    if last_location_of_definite_state > 0:
        
        if states[last_location_of_definite_state] == 1:
            states[last_location_of_definite_state:] = 2
        
        if states[last_location_of_definite_state] == 1:
            states[last_location_of_definite_state:] = 4
    
    # Set everywhere else as state 2:        
    states[states == 0] = 2
    
    if show:
        fig, ax = plt.subplots(figsize=(15, 4))
        t = np.linspace(0,round(len(pcg_signal)/sampling_rate),len(pcg_signal))
    
        ax.plot(t, pcg_signal, color='black')
        
        # arrange samples into groups of equal elements (split arrays into individual states)
        time_intervals = np.split(t, np.where(np.diff(states) != 0)[0]+1)
        states_intervals = np.split(states, np.where(np.diff(states) != 0)[0]+1)
        
        colors = ["green","red","pink","blue"]
        y_lims = ax.get_ylim()

        for i in range(len(time_intervals)):
            ax.fill_between(time_intervals[i], y_lims[0], y_lims[1], color=colors[int(states_intervals[i][0]-1)], alpha=0.4)       
        
        legend_elements = [Patch(facecolor=colors[0],edgecolor=colors[0],alpha = 0.4,label='S1'),
                           Patch(facecolor=colors[1],edgecolor=colors[1],alpha = 0.4,label='Systole'),
                           Patch(facecolor=colors[2],edgecolor=colors[2],alpha = 0.4,label='S2'),
                           Patch(facecolor=colors[3],edgecolor=colors[3],alpha = 0.4,label='Diastole')]

        ax.legend(handles=legend_elements,bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    return utils.ReturnTuple((states,), 
                             ('states',))
                             
