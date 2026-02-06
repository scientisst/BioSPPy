# -*- coding: utf-8 -*-
"""
biosppy.signals.egm
-------------------

This module provides methods to process intracardiac EGM (Electrogram) signals.

:copyright: (c) 2015-2016 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""
# 3rd party
import numpy as np
import scipy
import matplotlib.pyplot as plt
import inspect
import warnings

# local
from .. import plotting, utils


def egm(signal=None, sampling_rate=1000., type = 'bipolar', rhythm = None, woi=None, reference=None, method = 'nleo', threshold=None, show=True):
    """ Process intracardiac EGM (Electrogram) signals.
    The outputs of this function depend on the rhythm of the EGM signal.
    
    Parameters
    ----------
    signal : array
        An array with the EGM signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz) of the EGM signal (default is 1000 Hz).
    type : str, optional
        Type of EGM signal. Must be 'bipolar' or 'unipolar' (default is 'bipolar').
    rhythm : str
        Rhythm of the EGM signal. Must be 'sinus' or 'af'.
    woi : array, optional
        Window of interest as [from, to] in samples. If provided, the EGM signal will be cropped to this window.
    reference : int, optional
        Reference sample index for the window of interest. If provided, the window will be centered around this reference.
    method : str, optional
        Method to calculate activation time. Must be 'nleo', 'dvdt', 'max', or 'min' (default is 'nleo').
    threshold : float, optional
        Threshold for the NLEO signal. If provided, the NLEO signal will be thresholded.
    show : bool, optional
       If True, show a summary plot.
    
    Returns
    ----------
    ts : array
        Signal time axis reference (seconds). 
    filtered : array
        Filtered EGM signal.
    active_regions : array
        Active regions of the EGM signal.
    ts_windowed : array
        Time axis reference for the windowed EGM signal (seconds).
    windowed : array
        Windowed EGM signal.
    lat_index : int
        Index corresponding to the activation time of the EGM signal.
    lat : float
        Activation time in milliseconds.
    df : float
        Dominant frequency (Hz).
    freqs : array
        Frequencies of dominant frequency spectrum (Hz).
    power : array
        Power of dominant frequency spectrum.
    entropy : float
        Shannon entropy of the EGM signal.
    oi : float
        Organization index of the EGM signal.
    ri : array
        Regularity index of the EGM signal.
    """
    
    warnings.filterwarnings("ignore")
    
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")
    
    if type is None or type not in ['bipolar', 'unipolar']:
        raise ValueError("Invalid type. Must be 'bipolar' or 'unipolar'.")
    
    if rhythm is None or rhythm not in ['sinus', 'af']:
        raise ValueError("Invalid rhythm. Must be 'sinus' or 'af'.")
    
    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)
    
    # set outputs
    
    windowed = None
    ts_windowed = None
    lat_index = None
    lat = None
    df = None
    freqs = None
    spectrum = None
    entropy = None
    organization = None
    regularity = None
    
    # filter signal
    
    if type == 'bipolar':
        # bandpass filter for bipolar signals
        lowcut = 30.0
        highcut = 300.0
        order = 4
        b, a = scipy.signal.butter(order, [lowcut, highcut], btype='band', fs=sampling_rate)
        filtered_signal = scipy.signal.filtfilt(b, a, signal)
        
    else:
        # bandpass filter for unipolar signals
        lowcut = 0.05
        highcut = 600.0
        order = 4
        b, a = scipy.signal.butter(order, [lowcut, highcut], btype='band', fs=sampling_rate)
        filtered_signal = scipy.signal.filtfilt(b, a, signal)
        
    # get active regions
    _, _, nleo_threshold, _, _ = nleo(filtered_signal, sampling_rate=sampling_rate, plot=False)
    
    if rhythm == 'sinus':
        
        # calculate activation time using specified method
        methods = {
            'nleo': nleo_lat,
            'dvdt': dvdt_lat,
            'max': max_lat,
            'min': min_lat
        }
        
        if method not in methods:
            raise ValueError("Invalid method. Must be 'nleo', 'dvdt','max', or 'min'.")
        
        lat_index, lat = call_lat_method(lat_method=methods[method], filtered_signal=filtered_signal, sampling_rate=sampling_rate, woi=woi, reference=reference, plot=False)
        
    else:
        # for atrial fibrillation, calculate dominant frequency, entropy, organization index, and regularity index
        
        df, freqs, spectrum = dominant_frequency(filtered_signal, sampling_rate=sampling_rate, plot=False)
        
        entropy, = shannon_entropy(filtered_signal, sampling_rate=sampling_rate, plot=False)
        
        organization, = organization_index(filtered_signal, sampling_rate=sampling_rate)
        
        regularity, = regularity_index(filtered_signal, sampling_rate=sampling_rate)
        
        
    if woi is not None:
        windowed, = get_woi(signal, int(woi[0]), int(woi[1]), int(reference))
        ts_windowed = np.linspace(0, (len(windowed)-1)/sampling_rate, len(windowed), endpoint=True)
        
    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    
    args = [ts, filtered_signal, nleo_threshold, ts_windowed, windowed, lat_index, lat, df, freqs, spectrum, entropy, organization, regularity]
    names = ['ts', 'filtered', 'active_regions', 'ts_windowed', 'windowed', 'lat_index', 'lat', 'df', 'freqs', 'power', 'entropy', 'oi', 'ri']
        
    warnings.resetwarnings()
    
    if show:
        plotting.plot_egm(ts=ts,
             raw=signal,
             filtered=filtered_signal,
             rhythm=rhythm,
             active_regions=nleo_threshold,
             ts_windowed=ts_windowed,
             windowed = windowed,
             lat_index=lat_index,
             lat=lat,
             df = df,
             freqs = freqs,
             power = spectrum,
             entropy = entropy,
             oi = organization,
             ri = regularity,
             units=None,
             path=None,
             show=show)
        
    return utils.ReturnTuple(args, names)
        
def call_lat_method(lat_method, filtered_signal, sampling_rate, verbose=False, **kwargs):
    
    """Checks **kwargs against valid optional parameters for the selected method
    and calls it with the valid arguments.

    Parameters
    ----------
    lat_method : function
        The method to calculate activation time (e.g., 'nleo_activation_time').
    filtered_signal : array
        The preprocessed EGM signal to analyze.
    sampling_rate : float
        The sampling frequency of the EGM signal.
    woi : list, optional
        Window of interest as [from, to] in samples. If provided, the EGM signal will be cropped to this window.
    reference : int, optional
        Reference sample index for the window of interest. If provided, the window will be centered around this reference.
    plot : bool, optional
        If True, plots the EGM signal with activation time marked.
    **kwargs : dict
        Additional keyword arguments to pass to the activation time method.

    Returns
    -------
    lat_index : int
        Index corresponding to the activation time of the EGM signal.
    lat : float
        Activation time in milliseconds. If woi is provided, time is centered in the window.
    """
    
    sig = inspect.signature(lat_method)
    allowed_args = sig.parameters.keys()
    valid_args = {kwarg: val for kwarg, val in kwargs.items() if kwarg in allowed_args}
    if verbose:
        print(f"Passed valid parameters for activation method: {lat_method}: {valid_args}")
    lat_index, lat = lat_method(
        signal=filtered_signal, sampling_rate=sampling_rate, **valid_args
    )
    return (lat_index, lat)
    
    



def get_woi(signal, woi_from, woi_to, reference=None):
    """
    Extracts a window of interest (WOI) from the EGM signal.
    
    Parameters:
    ----------
    signal : array
        An array with the EGM signal.
    woi_from : int
        Starting index of the window of interest.
    woi_to : int
        Ending index of the window of interest.
    reference : int, optional
        Reference sample index for the window of interest. If provided, the window will be centered around this reference.
    
    Returns:
    ----------
    egm_woi : array
        The EGM signal cropped to the window of interest.
    """
    
    if reference is not None:
        idx_from = reference + woi_from
        idx_to = reference + woi_to
    else:
        idx_from = woi_from
        idx_to = woi_to
                
    return utils.ReturnTuple((signal[idx_from:idx_to],),
                             ('egm_woi',))

def nleo_lat(signal=None, sampling_rate=1000., woi = None, reference = None, plot=False):
    
    """
    Calculate the NLEO (Non-Linear Energy Operator) and activation times from EGM signals.
    
    Parameters:
    ----------
    egm : np.ndarray
        An array with the EGM signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    woi : list, optional
        Window of interest as [from, to] in samples. If provided, the EGM signal will be cropped to this window.
    reference : int, optional
        Reference sample index for the window of interest. If provided, the window will be centered around this reference.
    plot : bool, optional
        If True, plots the EGM signal with activation time marked.
    
    Returns:
    ----------
    nleo : np.ndarray
        The NLEO of the EGM signal.
    nleo_filt : np.ndarray
        The low-pass filtered NLEO.
    lat_index : np.ndarray
        Activation time corresponding to the EGM signal.
    """
    
    warnings.warn("Activation time can only be calculated for sinus rhythm signals.")
    
    _, _, _,lat_index, lat = nleo(signal, sampling_rate, woi=woi, reference=reference, plot=plot)
    
    return utils.ReturnTuple((lat_index, lat),
                             ('lat_index', 'lat'))
    

def nleo(signal=None, sampling_rate=1000., woi = None, reference = None, threshold = None, plot=False):
    """
    Calculate activation time based on the NLEO (Non-Linear Energy Operator) method.
    
    Parameters:
    ----------
    signal : array
        An array with the EGM signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz) of the EGM signal (default is 1000 Hz).
    woi : list, optional
        Window of interest as [from, to] in samples. If provided, the EGM signal will be cropped to this window.
    reference : int, optional
        Reference sample index for the window of interest. If provided, the window will be centered around this reference.
    plot : bool, optional
        If True, plots the EGM signal with activation time marked.
    
    Returns:
    ----------
    nleo : array
        The NLEO of the EGM signal.
    nleo_filt : array
        The low-pass filtered NLEO.
    nleo_threshold : array
        The thresholded NLEO signal.
    lat_index : int
        Index corresponding to the activation time of the EGM signal.
    lat : float
        Activation time in milliseconds. If woi is provided, time is centered in the window.
    """
    
    if woi is not None:
        signal, = get_woi(signal, int(woi[0]), int(woi[1]), int(reference))
            
    egm_squared = signal**2
    egm_rs = signal[1:]
    egm_ls = signal[:-1]
    
    # top and tail signals
    egm_squared = egm_squared[1:-1]
    egm_rs = egm_rs[1:]
    egm_ls = egm_ls[:-1]
    
    # calculate nleo
    nleo = egm_squared - (egm_rs * egm_ls)
    
    # add zeros to beginning and end
    nleo = np.concatenate(([0], nleo, [0]))

    # low-pass filter (zero phase)
    cutoff = 24 # Hz
    order = 8
    b, a = scipy.signal.butter(order, cutoff, btype='low', fs = sampling_rate)
    nleo_filt = scipy.signal.filtfilt(b, a, nleo)
    
    if threshold is None:
        P1 = np.percentile(abs(nleo_filt), 5)
        P2 = np.percentile(abs(nleo_filt), 95)
        threshold = P1 + 0.2 * (P2)
    nleo_threshold = np.zeros_like(nleo_filt)
    nleo_threshold[abs(nleo_filt) >= threshold] = 1
    
    # remove non-active parts shorter than 30 samples
    min_inactive_length = 30
    inactive_regions = np.where(nleo_threshold == 0)[0]
    diff_inactive = np.diff(inactive_regions)
    split_indices = np.where(diff_inactive > 1)[0]
    start_idx = 0
    for split_idx in split_indices:
        region = inactive_regions[start_idx:split_idx + 1]
        if len(region) < min_inactive_length:
            nleo_threshold[region] = 1
        start_idx = split_idx + 1
    # check last region
    region = inactive_regions[start_idx:]
    if len(region) < min_inactive_length:
        nleo_threshold[region] = 1
    
    # activation time 
    x = np.linspace(1, len(nleo_filt), len(nleo_filt))
    auc = scipy.integrate.cumulative_trapezoid(nleo_filt, x, initial=0)
    lat_index = np.interp(auc[-1] / 2, auc, x)
    
    # convert to time
    lat = lat_index / sampling_rate * 1000  # convert to milliseconds
    # center in window
    # if woi is not None:
    #     lat = lat - woi[0]  # center in window

    if plot:
        
        time = np.arange(len(signal)) / sampling_rate
        plt.plot(time, signal, label='Signal')
        plt.plot(time[int(lat_index)], signal[int(lat_index)], label='Activation Time', marker='o', markersize=8, color='red')
        plt.tight_layout()
        plt.show()
        
 
    return utils.ReturnTuple((nleo, nleo_filt, nleo_threshold, lat_index, lat),
                            ('nleo', 'nleo_filt', 'nleo_threshold', 'lat_index', 'lat'))

    

def dvdt_lat(signal=None, sampling_rate=1000., woi = None, reference = None, plot=False):
    
    """
    Calculate activation time based on maximum negative dv/dt from EGM signal.
    
    Parameters:
    ----------
    signal : array
        An array with the EGM signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz) of the EGM signal (default is 1000 Hz).
    woi : list, optional
        Window of interest as [from, to] in samples. If provided, the EGM signal will be cropped to this window.
    reference : int, optional
        Reference sample index for the window of interest. If provided, the window will be centered around this reference.
    plot : bool, optional
        If True, plots the EGM signal with activation time marked.
        
    Returns:
    ----------
    lat_index : int
        Index corresponding to the activation time of the EGM signal.
    lat : float
        Activation time in milliseconds. If woi is provided, time is centered in the window.
    """
    
    warnings.warn("Activation time can only be calculated for sinus rhythm signals.")

    if woi is not None:
        signal, = get_woi(signal, int(woi[0]), int(woi[1]), int(reference))
        
    # compute dv/dt
    dvdt = np.gradient(signal)
    
    # get activation time as index of maximum -dv/dt
    lat_index = np.argmax(-dvdt)
    
    # convert to time
    lat = lat_index / sampling_rate * 1000  # convert to milliseconds
    # center in window
    # if woi is not None:
    #     lat = lat - woi[0]  # center in window
    
    if plot:
        time = np.arange(len(signal)) / sampling_rate
        plt.plot(time, signal, label='Signal')
        plt.plot(time[lat_index], signal[lat_index], label='Activation Time', marker='o', markersize=8, color='red')
        plt.tight_layout()
        plt.show()

    return utils.ReturnTuple((lat_index, lat),
                             ('lat_index', 'lat'))



def max_lat(signal=None, sampling_rate=1000., woi = None, reference = None, plot=False):
    """
    Calculate activation time based on maximum amplitude from EGM signal.
    
    Parameters:
    ----------
    signal : array
        An array with the EGM signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz) of the EGM signal (default is 1000 Hz).
    woi : list, optional
        Window of interest as [from, to] in samples. If provided, the EGM signal will be cropped to this window.
    reference : int, optional
        Reference sample index for the window of interest. If provided, the window will be centered around this reference.
    plot : bool, optional
        If True, plots the EGM signal with activation time marked.
        
    Returns:
    ----------
    lat_index : int
        Index corresponding to the activation time of the EGM signal.
    lat : float
        Activation time in milliseconds. If woi is provided, time is centered in the window.
    """
    
    warnings.warn("Activation time can only be calculated for sinus rhythm signals.")

    if woi is not None:
        signal, = get_woi(signal, int(woi[0]), int(woi[1]), int(reference))
        
    # get activation time as index of maximum amplitude
    lat_index = np.argmax(signal)
    
    # convert to time
    lat = lat_index / sampling_rate * 1000  # convert to milliseconds
    # center in window
    # if woi is not None:
    #     lat = lat - woi[0]  # center in window
    
    if plot:
        time = np.arange(len(signal)) / sampling_rate
        plt.plot(time, signal, label='EGM Signal')
        plt.plot(time[lat_index], signal[lat_index], label='Activation Time', marker='o', markersize=8, color='red')
        plt.tight_layout()
        plt.show()

    return utils.ReturnTuple((lat_index, lat),
                             ('lat_index', 'lat'))

def min_lat(signal=None, sampling_rate=1000., woi = None, reference = None, plot=False):
    """
    Calculate activation time based on minimum amplitude from EGM signal.
    
    Parameters:
    ----------
    signal : array
        An array with the EGM signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz) of the EGM signal (default is 1000 Hz).
    woi : list, optional
        Window of interest as [from, to] in samples. If provided, the EGM signal will be cropped to this window.
    reference : int, optional
        Reference sample index for the window of interest. If provided, the window will be centered around this reference.
    plot : bool, optional
        If True, plots the EGM signal with activation time marked.
        
    Returns:
    ----------
    lat_index : int
        Index corresponding to the activation time of the EGM signal.
    lat : float
        Activation time in milliseconds. If woi is provided, time is centered in the window.
    """
    
    warnings.warn("Activation time can only be calculated for sinus rhythm signals.")

    if woi is not None:
        signal, = get_woi(signal, int(woi[0]), int(woi[1]), int(reference))
        
    # get activation time as index of minimum amplitude
    lat_index = np.argmin(signal)
    
    # convert to time
    lat = lat_index / sampling_rate * 1000  # convert to milliseconds
    # center in window
    # if woi is not None:
    #     lat = lat - woi[0]  # center in window
    
    if plot:
        time = np.arange(len(signal)) / sampling_rate
        plt.plot(time, signal, label='EGM Signal')
        plt.plot(time[lat_index], signal[lat_index], label='Activation Time', marker='o', markersize=8, color='red')
        plt.tight_layout()
        plt.show()

    return utils.ReturnTuple((lat_index, lat),
                             ('lat_index', 'lat'))

def get_activation_times(signals=None, sampling_rate=1000., woi=None, reference=None, method='nleo'):
    """
    Calculate activation times for multiple EGM signals.
    
    Parameters:
    ----------
    signals : array
        A 2D array where each row is an EGM signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz) of the EGM signals (default is 1000 Hz).
    woi : list, optional
        Window of interest as [from, to] in samples. If provided, each EGM signal will be cropped to this window.
    reference : int, optional
        Reference sample index for the window of interest. If provided, the window will be centered around this reference.
    
    Returns:
    lat_indexes : int
        Indexes corresponding to the activation times of each EGM signal.
    lats : float
        Activation times in milliseconds. If woi is provided, time is centered in the window.
    """
    
    warnings.warn("Activation time can only be calculated for sinus rhythm signals.")

    n_signals = signals.shape[0]
    lat_idx = np.zeros(n_signals)
    lat_times = np.zeros(n_signals)
    
    for i in range(n_signals):
        if method == 'nleo':
            _, _, lat_index, lat = nleo_lat(signals[i, :], sampling_rate, woi=woi, reference=reference)
        elif method == 'dvdt':
            lat_index, lat = dvdt_lat(signals[i, :], sampling_rate, woi=woi, reference=reference)
        elif method == 'max':
            lat_index, lat = max_lat(signals[i, :], sampling_rate, woi=woi, reference=reference)
        elif method == 'min':
            lat_index, lat = min_lat(signals[i, :], sampling_rate, woi=woi, reference=reference)
        else:
            raise ValueError("Invalid method. Must be 'nleo', 'dvdt','max', or 'min'.")
        lat_idx[i] = lat_index
        lat_times[i] = lat
    return utils.ReturnTuple((lat_idx, lat_times),
                             ('lat_indexes', 'lats'))

def compare_activation_times(signal=None, sampling_rate=1000., woi=None, reference=None, plot=False):
    """
    Compare activation times calculated by different methods for a single EGM signal.
    
    Parameters:
    ----------
    signal : array
        An array with the EGM signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz) of the EGM signal (default is 1000 Hz).
    woi : list, optional
        Window of interest as [from, to] in samples. If provided, the EGM signal will be cropped to this window.
    reference : int, optional
        Reference sample index for the window of interest. If provided, the window will be centered around this reference.
    plot : bool, optional
        If True, plots the EGM signal with activation times marked.
    
    Returns:
    ----------
    lat_index_nleo : int
        Index corresponding to the activation time from NLEO method.
    lat_index_dvdt : int
        Index corresponding to the activation time from -dv/dt method.
    lat_index_max : int
        Index corresponding to the activation time from maximum amplitude method.
    lat_index_min : int
        Index corresponding to the activation time from minimum amplitude method.
    lat_nleo : float
        Activation time in milliseconds from NLEO method.
    lat_dvdt : float
        Activation time in milliseconds from -dv/dt method.
    lat_max : float
        Activation time in milliseconds from maximum amplitude method.
    lat_min : float
        Activation time in milliseconds from minimum amplitude method.
    """
    
    warnings.warn("Activation time can only be calculated for sinus rhythm signals.")
    
   # run all methods
    lat_index_nleo, lat_nleo = nleo_lat(signal, sampling_rate, woi=woi, reference=reference)
    lat_index_dvdt, lat_dvdt = dvdt_lat(signal, sampling_rate, woi=woi, reference=reference)
    lat_index_max, lat_max = max_lat(signal, sampling_rate, woi=woi, reference=reference)
    lat_index_min, lat_min = min_lat(signal, sampling_rate, woi=woi, reference=reference)
    
    if plot:
        if woi is not None:
            signal, = get_woi(signal, int(woi[0]), int(woi[1]), int(reference))
        time = np.arange(len(signal)) / sampling_rate
        plt.figure()
        plt.plot(time, signal, label='Signal')
        plt.plot(time[int(lat_index_nleo)], signal[int(lat_index_nleo)], label='NLEO Activation Time', marker='o', markersize=8)
        plt.plot(time[int(lat_index_dvdt)], signal[int(lat_index_dvdt)], label='-dv/dt Activation Time', marker='o', markersize=8)
        plt.plot(time[int(lat_index_max)], signal[int(lat_index_max)], label='Max Amplitude Activation Time', marker='o', markersize=8)
        plt.plot(time[int(lat_index_min)], signal[int(lat_index_min)], label='Min Amplitude Activation Time', marker='o', markersize=8)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return utils.ReturnTuple((lat_index_nleo, lat_index_dvdt, lat_index_max, lat_index_min, lat_nleo, lat_dvdt, lat_max, lat_min),
                             ('lat_index_nleo', 'lat_index_dvdt', 'lat_index_max', 'lat_index_min', 'lat_nleo', 'lat_dvdt', 'lat_max', 'lat_min'))

def get_voltage(signal=None, woi=None, reference=None):
    """
    Get the maximum voltage of the EGM signal.
    
    Parameters:
    ----------
    signal : array
        An array with the EGM signal.
    woi : list, optional
        Window of interest as [from, to] in samples. If provided, the EGM signal will be cropped to this window.
    reference : int, optional
        Reference sample index for the window of interest. If provided, the window will be centered around this reference.
    
    Returns:
    ----------
    voltage : float
        The maximum voltage of the EGM signal.
    """
    warnings.filterwarnings("ignore")

    idx, _ = max_lat(signal, woi=woi, reference=reference)
    
    warnings.resetwarnings()
    
    if woi is not None:
        signal, = get_woi(signal, int(woi[0]), int(woi[1]), int(reference))
    voltage = signal[int(idx)]
    
    
    
    return utils.ReturnTuple((voltage,),
                             ('voltage',))

def get_voltages(signals=None, woi=None, reference=None):
    """
    Get the maximum voltages for multiple EGM signals.
    
    Parameters:
    ----------
    signals : array
        A 2D array where each row is an EGM signal.
    woi : list, optional
        Window of interest as [from, to] in samples. If provided, each EGM signal will be cropped to this window.
    reference : int, optional
        Reference sample index for the window of interest. If provided, the window will be centered around this reference.
    
    Returns:
    ----------
    voltages : array
        An array of maximum voltages for each EGM signal.
    """
    
    n_signals = signals.shape[0]
    voltages = np.zeros(n_signals)
    
    for i in range(n_signals):
        voltage_temp = get_voltage(signals[i, :], woi=woi, reference=reference)
        voltages[i] = voltage_temp
        
    return utils.ReturnTuple((voltages,),
                             ('voltages',))

def dominant_frequency(signal=None, sampling_rate=1000., plot=False):
    """
    Get the dominant frequency of the EGM signal.
    
    Parameters:
    ----------
    signal : array
        An array with the EGM signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz) of the EGM signal (default is 1000 Hz).
    plot : bool, optional
        If True, plots the power spectral density of the EGM signal.
    
    Returns:
    ----------
    dominant_freq : float
        The dominant frequency of the EGM signal in Hz.
    fft_freqs : array
        The frequencies of the power spectral density of the dominant frequency spectrum.
    psd : array
        The power spectral density of the dominant frequency spectrum.
    """
    warnings.warn("Dominant frequency can only be calculated for atrial fibrillation signals.")
        
    # Butterworth IIR bandpass filter between 40 and 250 Hz
    lowcut = 40.0
    highcut = 250.0
    order = 8
    b, a = scipy.signal.butter(order, [lowcut, highcut], btype='band', fs=sampling_rate)
    signal = scipy.signal.filtfilt(b, a, signal)
        
    # signal rectification
    signal = np.abs(signal)
    
    # Butterworth IIR low-pass filter with cutoff frequency of 20 Hz
    cutoff = 20.0
    b, a = scipy.signal.butter(order, cutoff, btype='low', fs=sampling_rate)
    signal = scipy.signal.filtfilt(b, a, signal)
    
    # compute fft with zero padding to next power of 2
    n = len(signal)
    nfft = 2**np.ceil(np.log2(n)).astype(int)
    # pad signal
    signal = np.pad(signal, (0, nfft - n), mode='constant')
    from scipy.signal.windows import hann
    from scipy.fft import fft, fftfreq
    w = hann(nfft)
    fft_vals = fft(signal*w)
    
    # sample spacing
    T = 1.0 / sampling_rate
    fft_freqs = fftfreq(nfft, T)[:nfft // 2]
    fft_vals = fft_vals[:nfft // 2]
    psd = 2.0 / (nfft * T) * np.abs(fft_vals)**2
    
    # get dominant frequency between 3 and 10 Hz
    idx = np.where((fft_freqs >= 3) & (fft_freqs <= 10))
    psd_max = psd[idx]
    fft_freqs_max = fft_freqs[idx]
    dominant_freq = fft_freqs_max[np.argmax(psd_max)]
    
    if plot:
        plt.figure()
        plt.plot(fft_freqs, psd)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.axvline(dominant_freq, color='r', linestyle='--', label='Dominant Frequency = ' + '%.2f' % dominant_freq + " Hz")
        plt.legend()
        plt.xlim(0, 20)
        plt.show()
        
    
    return utils.ReturnTuple((dominant_freq, fft_freqs, psd),
                             ('dominant_freq', 'fft_freqs', 'psd'))

def shannon_entropy(signal=None, sampling_rate=1000., plot=False):
    """
    Calculate Shannon entropy of the EGM signal.
    
    Parameters:
    ----------
    signal : array
        An array with the EGM signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz) of the EGM signal (default is 1000 Hz).
    plot : bool, optional
        If True, plots the histogram of the EGM signal voltages.
    
    Returns:
    ----------
    shannon_entropy : float
        Shannon entropy of the EGM signal.
    """
        
    # histogram with binsize of 0.01 mV
    bins = np.arange(np.min(signal), np.max(signal), 0.01)
    counts, bins = np.histogram(signal, bins=bins)
    
    # calculate Shannon entropy. If counts = 0, probability = 0
    probabilities = counts / np.sum(counts)
    probabilities = probabilities[probabilities != 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    if plot:
        plt.figure()
        plt.stairs(counts, bins)
        plt.xlabel('Bins')
        plt.ylabel('Probability')
        plt.text(np.mean(bins), np.max(counts), 'Entropy = ' + '%.2f' % entropy)
        plt.show()
    
    return utils.ReturnTuple((entropy,),
                             ('shannon_entropy',))


def organization_index(signal=None, sampling_rate=1000.):
    """
    Calculate organization index of the EGM signal.
    
    Parameters:
    ----------
    signal : array
        An array with the EGM signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz) of the EGM signal (default is 1000 Hz).
    
    Returns:
    ----------
    organization_index : float
        Organization index of the EGM signal.
    """
    
    df, freqs, spectrum = dominant_frequency(signal, sampling_rate=sampling_rate, plot=False)
    
    if df >= 10:
        raise ValueError("Dominant frequency is greater than 10 Hz. Organization index cannot be calculated.")
    
    # get area under power spectrum for df plus and minus 0.75 Hz
    idx = np.where((freqs >= df - 0.75) & (freqs <= df + 0.75))
    spectrum1 = spectrum[idx]
    area1 = scipy.integrate.trapezoid(spectrum1)
    
    # get area under power spectrum for first harmonic of df plus and minus 0.75 Hz
    idx = np.where((freqs >= 2 * df - 0.75) & (freqs <= 2 * df + 0.75))
    spectrum2 = spectrum[idx]
    area2 = scipy.integrate.trapezoid(spectrum2)
    
    # get area under power spectrum for second harmonic of df plus and minus 0.75 Hz
    idx = np.where((freqs >= 3 * df - 0.75) & (freqs <= 3 * df + 0.75))
    spectrum3 = spectrum[idx]
    area3 = scipy.integrate.trapezoid(spectrum3)
    
    # get area between 3 and 20 Hz
    idx = np.where((freqs >= 3) & (freqs <= 20))
    spectrum_total = spectrum[idx]
    area_total = scipy.integrate.trapezoid(spectrum_total)
    
    # calculate organization index
    oi = (area1 + area2 + area3) / area_total
    
    return utils.ReturnTuple((oi,),
                             ('organization_index',))

def regularity_index(signal=None, sampling_rate=1000.):
    """
    Calculate regularity index of the EGM signal.
    
    Parameters:
    ----------
    signal : array
        An array with the EGM signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz) of the EGM signal (default is 1000 Hz).
    
    Returns:
    ----------
    regularity_index : float
        Regularity index of the EGM signal.
    """
    
    df, freqs, spectrum = dominant_frequency(signal, sampling_rate=sampling_rate, plot=False)
    
    if df >= 10:
        raise ValueError("Dominant frequency is greater than 10 Hz. Regularity index cannot be calculated.")
    
    # get area under power spectrum for df plus and minus 0.75 Hz
    idx = np.where((freqs >= df - 0.75) & (freqs <= df + 0.75))
    spectrum1 = spectrum[idx]
    area1 = scipy.integrate.trapezoid(spectrum1)
    
    # get area between 3 and 20 Hz
    idx = np.where((freqs >= 3) & (freqs <= 20))
    spectrum_total = spectrum[idx]
    area_total = scipy.integrate.trapezoid(spectrum_total)
    
    # calculate regularity index
    ri = area1 / area_total
    
    return utils.ReturnTuple((ri,),
                             ('regularity_index',))