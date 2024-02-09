# -*- coding: utf-8 -*-
"""
biosppy.synthesizers.emg
-------------------
This module provides methods to synthesize Electromyographic (EMG) signals.
:copyright: (c) 2015-2021 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
import numbers

# 3rd party
import numpy as np
import warnings
import matplotlib.pyplot as plt

# local
from .. import plotting, utils


def synth_uniform(duration=10, 
                  length=None, 
                  sampling_rate=1000, 
                  noise=0.01,
                  baseline = None,
                  burst_number=1, 
                  burst_duration=1.0,
                  burst_location = None,
                  amplitude_mult = None, 
                  random_state=None
                  ): # Default values
    
    """Generates an artificial (synthetic) EMG signal of a given duration and sampling rate, with muscle activity bursts modeled 
    as an uniform distribution and background noise modeled as a zero-mean Gaussian process with adjustable standard deviation. 

    Follows the approach by Diong, Joanna [ModelEMG1], but, additionally, this function also allows to manually choose the muscle 
    activity burst locations, and add amplitude multipliers for each burst.

    If the parameters introduced lead to superimposed burst locations, an error will be raised, and if they lead to consecutive
    bursts, a warning will be raised. Warnings will also be raised, if the precision derived from the selected sampling rate is not
    compatible with each burst's location or duration, or the duration of quiet periods.

    Parameters
    ----------
    duration : int, optional
        Desired recording length in seconds.
    sampling_rate : int, optional
        The desired sampling rate (in Hz, i.e., samples/second).
    length : int, optional
        The desired length of the signal (in samples).
    noise : float, optional
        Noise level (standard deviation of the gaussian distribution from which values are sampled).
    baseline : float, optional
        Signal offset from zero. If no value is given, it is assumed that the baseline has already been removed.
    burst_number : int, optional
        Desired number of bursts of activity (active muscle periods).
    burst_duration : float or list, optional
        Duration of the bursts. Can be a float (each burst will have the same duration) or a list of durations for 
        each burst.
    burst_location : list, optional
        Location of the bursts (in seconds). 
    amplitude_mult : float or list, optional
        Amplitude multiplier for the bursts. Can be a float (each burst will have the same amplitude range) or a 
        list of multipliers for each bursts.
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator, optional
        Seed for the random number generator. 

    Returns
    ----------
    emg : array
        Vector containing the EMG signal.
    t : array
        Time values accoring to the provided sampling rate.
    params : dict
        Input parameters of the function, clean EMG and noise signals, and SNR

    Examples
    ----------
    sampling_rate = 1000
    duration = 10
    noise_amplitude = 0.05
    bursts = 7
    burst_duration = [0.5,1,0.5,0.6,1,0.5,0.5]
    burst_location = [0.1,2.5,4,5.5,7,8.5,9.4]
    amplitude_mult = [1,1,0.5,1.5,1,0.75,1]
    emg_synth, t, params = synth_uniform(duration=duration, sampling_rate=sampling_rate, noise=noise_amplitude, 
                                        burst_number=bursts, burst_duration=burst_duration, burst_location=burst_location,
                                        amplitude_mult=amplitude_mult)
    
    # Get muscle activity state
    activity = params["activity"]

    plt.plot(t,emg_synth,label="EMG")
    plt.plot(t,activity,label="Muscle activity")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid()
    plt.title("EMG")
    plt.legend()

    plt.show()

    References
    -----------
    .. [ModelEMG1] Joanna DIONG, 
    "PYTHON: ANALYSING EMG SIGNALS",
    https://scientificallysound.org/2016/08/11/python-analysing-emg-signals-part-1/
    """
    # Seed the random generator for reproducible results
    # If seed is an integer, use the legacy RandomState class
    if isinstance(random_state, numbers.Integral):
        rng = np.random.RandomState(random_state)
    # If seed is already a random number generator class return it as it is
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rng = random_state
    # If seed is something else, use the new Generator class
    else:
        rng = np.random.default_rng(random_state) 

    if not isinstance(sampling_rate, (int, float)):
        raise TypeError("Error! 'sampling_rate' value must be an integer or float.")
    if sampling_rate <= 0:
        raise ValueError("Error! 'sampling_rate' value must be positive.")
    if sampling_rate < 100:
        warnings.warn("Sampling rate values below 100 Hz might lead to signals where onsets cannot be detected properly.")
    if sampling_rate < 1000:
        warnings.warn("Sampling rate values below 1000 Hz will not allow to effectively capture all significant frequency components of the EMG signal.")

    if not isinstance(burst_number, int):
        raise TypeError("Error! 'burst_number' value must be an integer.")
    if burst_number <= 0:
        raise ValueError("Error! 'burst_number' value must be positive.")
    
    if not isinstance(noise, (int, float)):
        raise TypeError("Error! 'noise' value must be an integer or float.")
    if noise < 0:
        raise ValueError("Error! 'noise' value must be non-negative.")
    
    if not isinstance(duration, (int, float)):
        raise TypeError("Error! 'duration' value must be an integer or float.")
    if duration <= 0:
        raise ValueError("Error! 'duration' value must be positive.")
    
    # Generate number of samples automatically if length is unspecified
    if length is None:
        length = duration * sampling_rate + 1
    else:
        if not isinstance(length, (int, float)):
            raise TypeError("Error! 'length' value must be an integer or float.")
        if length != duration * sampling_rate + 1:
            raise ValueError("Error! Signal length does not match duration with the given sampling rate.")

    if baseline is None:
        baseline = 0
    if not isinstance(baseline, (int, float)):
        raise TypeError("Error! Baseline value must be an integer or float.")
    
    if isinstance(burst_duration, (int, float)):
        burst_duration = np.repeat(burst_duration, burst_number)
    if not all((np.round((dur)/(1/sampling_rate),10)).is_integer() for dur in burst_duration):
        warnings.warn("The precision derived from the selected sampling rate is not compatible with the burst duration. This might alter or compromise synthesization of the signals, increasing the sampling rate is recommended.")
    if len(burst_duration) != burst_number:
        raise ValueError("Error! 'burst_duration' cannot be longer than the value of 'burst_number'.")

    total_duration_bursts = np.sum(burst_duration)
    if total_duration_bursts > duration:
        raise ValueError("Error! The total duration of bursts cannot exceed the total duration of the signal.")
    
    # Number of quiet periods (in between bursts)
    n_quiet = burst_number + 1  
    quiet_duration = np.repeat((duration - total_duration_bursts) / n_quiet, n_quiet)
    if burst_location is not None:
        if not (isinstance(burst_location, list) and all((isinstance(loc, int) or isinstance(loc, float)) for loc in burst_location)):
            raise TypeError("Error! 'burst_location' must be a list of integers or floats.")
        if len(burst_location) != burst_number:
            raise ValueError("Error! 'burst_location' list cannot be longer than the value of 'burst_number'.")
        if not all(np.round(((loc)/(1/sampling_rate)),10).is_integer() for loc in burst_location):
            warnings.warn("The precision derived from the selected sampling rate is not compatible with the burst locations. This might alter or compromise synthesization of the signals, increasing the sampling rate or altering the burst location/duration is recommended.")
        burst_location.sort()

        consecutive_bursts = False
        for i in range(len(burst_location)):
            loc = burst_location[i]
            dur = burst_duration[i]
            if loc < 0 or loc + dur > duration:
                raise ValueError("Error! Burst location times must be numbers greater than 0, and summed with their duration cannot be greater than the signal duration.")
            else:
                if i == 0: # First burst
                    quiet_duration[i] = loc
                    if burst_number == 1:
                        quiet_duration[i+1] = duration - (loc + dur)
                    else:
                        next_loc = burst_location[i+1]
                        if loc + dur >= next_loc: 
                            raise ValueError("Error! Burst location times overlap.")
                        elif loc + dur == next_loc - 1/sampling_rate:
                            consecutive_bursts = True
                elif i < burst_number - 1: # Intermediate burst
                    next_loc = burst_location[i+1]
                    if loc + dur >= next_loc: 
                            raise ValueError("Error! Burst location times overlap.")
                    elif loc + dur == next_loc - 1/sampling_rate:
                        consecutive_bursts = True
                    prev_loc = burst_location[i-1]
                    prev_dur = burst_duration[i-1]
                    quiet_duration[i] = loc - (prev_loc + prev_dur)
                else: # Final burst
                    prev_loc = burst_location[i-1]
                    prev_dur = burst_duration[i-1]
                    quiet_duration[i] = loc - (prev_loc + prev_dur)
                    quiet_duration[i+1] = duration - (loc + dur)
        if consecutive_bursts:
            warnings.warn("The defined burst locations and respective duration of each have lead to consecutive muscle activity bursts. This might hinder the distinction between bursts.")
    else:
        if not all(np.round((dur)/(1/sampling_rate),10).is_integer() for dur in quiet_duration):
            warnings.warn("The precision derived from the selected sampling rate is not compatible with the duration of quiet periods between evenly distributed bursts. This might alter (unequal duration of quiet periods) or compromise (impossible to separate bursts in the time domain) synthesization of the signals.")
    total_duration = total_duration_bursts + np.sum(quiet_duration)
    if np.round(total_duration,10) != duration:
        raise ValueError("Error! The total duration of bursts and quiet periods does not match the total duration of the signal.")
    
    if amplitude_mult is None:
        amplitude_mult = 1
    if isinstance(amplitude_mult, (int, float)):
        amplitude_mult = [amplitude_mult] * burst_number
    if not ((isinstance(amplitude_mult, list))and all((isinstance(amp, int) or isinstance(amp, float)) for amp in amplitude_mult)):
        raise TypeError("Error! 'amplitude_mult' must be an integer or float, or a list of this type.")
    if len(amplitude_mult) != burst_number:
        raise ValueError("Error! 'amplitude_mult' cannot be longer than the value of 'burst_number'.")

    # Generate bursts
    bursts = []
    curr_samples = 0
    for burst in range(burst_number):
        size = int(sampling_rate * burst_duration[burst]+1)
        bursts += [list(rng.uniform(-1*amplitude_mult[burst], 1*amplitude_mult[burst], size=size) + baseline)]
        curr_samples += size

    # Generate quiet periods
    quiets = []
    for quiet in range(n_quiet):  
        if quiet == 0:
            size = int(round(sampling_rate * quiet_duration[quiet],0))
        elif quiet != n_quiet - 1:
            size = int(round(sampling_rate * quiet_duration[quiet]-1,0))
        else:
            size = int(round(sampling_rate * quiet_duration[quiet]-1,0))
            # Guarantee signal length is respected (could be more or less samples due to rounding off errors, when using a sampling 
            # rate where the derived precision is not compatible with burst locations/duration or the duration of quiet periods):
            if length != curr_samples+size:
                size = length-curr_samples 
        curr_samples += size
        quiets += [list(np.zeros((size,)) + baseline)]

    # Merge muscle activity bursts and quiet periods
    emg = []
    activity = []
    for i in range(len(quiets)):  
        emg += quiets[i]
        activity += [list(np.zeros((len(quiets[i]),)))]
        if i < len(bursts):
            emg += bursts[i]
            activity += [list(np.ones((len(bursts[i]),)))]
    activity = np.concatenate(activity)
    emg = np.array(emg)

    # Add random (gaussian distributed) noise
    noise_signal = rng.normal(0, noise, length)
    max_noise = max(noise_signal)
    if any(max_noise > amplitude_mult):
        warnings.warn("Maximum burst amplitude is smaller than the maximum level of noise.")
    SNR = 20*np.log10(np.sqrt(np.mean(emg**2))/np.sqrt(np.mean(noise_signal**2)))
    emg_synth = emg + noise_signal

    t = np.arange(0,duration+1/sampling_rate,1/sampling_rate)

    params = {"duration": duration, 
              "length": length, 
              "sampling_rate": sampling_rate, 
              "noise": noise,
              "baseline": baseline,
              "burst_number": burst_number,
              "burst_duration": burst_duration,
              "burst_location": burst_location,
              "amplitude_mult": amplitude_mult,
              "random_state": rng,
              "clean_signal": emg,
              "noise_signal": noise_signal,
              "SNR": SNR,
              "activity": activity}

    args = (emg_synth, t, params)
    names = ("emg", "t", "params")

    return utils.ReturnTuple(args, names)


def _truncated_gaussian_window(sigma,
                               alpha,
                               sampling_rate):
    """Generates a truncated Gaussian window with duration (in seconds) given by 2*sigma*alpha

    Follows the approach by Ghislieri, Cerone, Knaflitz and Agostini [ModelEMG2].

    If the parameters introduced don't make sense in this context, an error will raise.

    Parameters
    ----------
    sigma : float
        Standard deviation of the truncated Gaussian process with zero mean.
    alpha : float
        Multiplier that multiplied with 'sigma' defines the time support of the truncated Gaussian process.
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).

    Returns
    -------
    win : array
        The truncated Gaussian window.

    References
    -----------
    .. [ModelEMG] Marco GHISLIERI, Giacinto Luigi CERONE, Marco KNAFLITZ & Valentina AGOSTINI
       "LONG SHORT-TERM MEMORY (LSTM) RECURRENT NEURAL NETWORK FOR MUSCLE ACTIVITY DETECTION"
       Journal of NeuroEngineering and Rehabilitation, Vol. 18, No. 1, 2021, 3–4
    """
    if not isinstance(sigma, (int, float)):
        raise TypeError("Error! 'sigma' value must be an integer or float.")
    if sigma <= 0:
        raise ValueError("Error! 'sigma' value must be positive.")
    
    if not isinstance(alpha, (int, float)):
        raise TypeError("Error! 'alpha' value must be an integer or float.")
    if alpha <= 0:
        raise ValueError("Error! 'alpha' value must be positive.")

    dur = 2*sigma*alpha
    x = np.arange(np.round(-dur/2,5), np.round(dur/2+1/sampling_rate,5), 1/sampling_rate)
    win = np.exp(-(x**2)/(2*sigma**2))

    return win
    

def synth_gaussian(duration=10,
                   sampling_rate=1000, 
                   length=None,
                   SNR=30,
                   sigma=0.1,
                   alpha=2.5,
                   baseline=None,
                   burst_number=1,
                   burst_location=None,
                   random_state=None):
    
    """Generates an artificial (synthetic) EMG signal of a given duration and sampling rate.

    Follows the approach by by Ghislieri, Cerone, Knaflitz and Agostini [ModelEMG2], where muscle activity bursts are modeled as a 
    zero-mean Gaussian process with standard deviation equal to 10**(SNR/20) mV, and the background noise is modeled as a zero-mean 
    Gaussian process with standard deviation equal to 1 mV. All muscle activity bursts have the same duration, which is equal to 
    2*sigma*alpha (in seconds).

    If the parameters introduced lead to superimposed burst locations, an error will be raised, and if they lead to consecutive
    bursts, a warning will raise. Warnings will also be raised, if the precision derived from the selected sampling rate is not
    compatible with each burst's location or duration, or the duration of quiet periods.

    Parameters
    ----------
    duration : int, optional
        Desired recording length in seconds.
    sampling_rate : int, optional
        The desired sampling rate (in Hz, i.e., samples/second).
    length : int, optional, optional
        The desired length of the signal (in samples).
    SNR : float, optional
        Desired signal-to-noise ratio of the signal (in dB).
    sigma : float, optional
        Standard deviation of the truncated Gaussian process with zero mean used to simulate muscle activity.
    alpha : float, optional
        Multiplier that multiplied with the 'sigma' value defines the time support of the truncated Gaussian process, which
        will be the duration of the bursts (burst_duration = 2*sigma*alpha, with default values, this duration is 0.5s).
    baseline : float, optional
        Signal offset from zero. If no value is given, it is assumed that the baseline has already been removed.
    burst_number : int, optional
        Desired number of bursts of activity (active muscle periods).
    burst_location : list, optional
        Location of the bursts (in seconds). 
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator. 

    Returns
    ----------
    emg : array
        Vector containing the EMG signal.
    t : array
        Time values accoring to the provided sampling rate.
    params : dict
        Input parameters of the function, clean EMG and noise signals, and SNR

    Examples
    ----------
    sampling_rate = 1000
    duration = 10
    SNR = 30
    sigma = 0.2
    alpha = 1.25
    bursts = 4
    burst_location = [2,4,6,8]
    output = synth_gaussian(duration=duration, sampling_rate=sampling_rate, SNR=SNR, 
                                    sigma=sigma, alpha=alpha, burst_number=bursts, burst_location=burst_location,
                                    random_state=0)
    emg_synth, t, params = output["emg"], output["t"], output["params"]

    # Get muscle activity state
    activity = params["activity"]

    plt.figure()
    plt.plot(t,emg_synth,label="EMG")
    plt.plot(t,activity,label="Muscle activity")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid()
    plt.title("EMG")
    plt.legend()

    plt.show()

    References
    -----------
    .. [ModelEMG2] Marco GHISLIERI, Giacinto Luigi CERONE, Marco KNAFLITZ & Valentina AGOSTINI
       "LONG SHORT-TERM MEMORY (LSTM) RECURRENT NEURAL NETWORK FOR MUSCLE ACTIVITY DETECTION"
       Journal of NeuroEngineering and Rehabilitation, Vol. 18, No. 1, 2021, 3–4
    """
    # Seed the random generator for reproducible results
    # If seed is an integer, use the legacy RandomState class
    if isinstance(random_state, numbers.Integral):
        rng = np.random.RandomState(random_state)
    # If seed is already a random number generator class return it as it is
    elif isinstance(random_state, (np.random.Generator, np.random.RandomState)):
        rng = random_state
    # If seed is something else, use the new Generator class
    else:
        rng = np.random.default_rng(random_state) 

    if not isinstance(sampling_rate, (int, float)):
        raise TypeError("Error! 'sampling_rate' value must be an integer or float.")
    if sampling_rate <= 0:
        raise ValueError("Error! 'sampling_rate' value must be positive.")
    if sampling_rate < 100:
        warnings.warn("Sampling rate values below 100 Hz might lead to signals where onsets cannot be detected properly.")
    if sampling_rate < 1000:
        warnings.warn("Sampling rate values below 1000 Hz will not allow to effectively capture all significant frequency components of the EMG signal.")

    if not isinstance(burst_number, int):
        raise TypeError("Error! 'burst_number' value must be an integer.")
    if burst_number <= 0:
        raise ValueError("Error! 'burst_number' value must be positive.")
    
    if not isinstance(SNR, (int, float)):
        raise TypeError("Error! 'SNR' value must be an integer or float.")
    
    if not isinstance(duration, (int, float)):
        raise TypeError("Error! 'duration' value must be an integer or float.")
    if duration <= 0:
        raise ValueError("Error! 'duration' value must be positive.")
    
    # Generate number of samples automatically if length is unspecified
    if length is None:
        length = int(duration * sampling_rate + 1)
    else:
        if not isinstance(length, (int, float)):
            raise TypeError("Error! 'length' value must be an integer or float.")
        if length != duration * sampling_rate + 1:
            raise ValueError("Error! Signal length does not match duration with the given sampling rate.")

    # Checking for invalid inputs
    if baseline is None:
        baseline = 0
    if not isinstance(baseline, (int, float)):
        raise TypeError("Error! Baseline value must be an integer or float.")
    
    if not isinstance(sigma, (int, float)):
        raise TypeError("Error! 'sigma' value must be an integer or float.")
    if sigma <= 0:
        raise ValueError("Error! 'sigma' value must be positive.")
    
    if not isinstance(alpha, (int, float)):
        raise TypeError("Error! 'alpha' value must be an integer or float.")
    if alpha <= 0:
        raise ValueError("Error! 'alpha' value must be positive.")
    
    burst_duration = 2*alpha*sigma 
    if not (np.round((burst_duration)/(1/sampling_rate),10)).is_integer():
        warnings.warn("The precision derived from the selected sampling rate is not compatible with the burst duration. This might alter or compromise synthesization of the signals, increasing the sampling rate is recommended.")
    if isinstance(burst_duration, (int, float)):
        burst_duration = np.repeat(burst_duration, burst_number)
    if len(burst_duration) != burst_number:
        raise ValueError("Error! 'burst_duration' list cannot be longer than the value of 'burst_number'.")

    total_duration_bursts = np.sum(burst_duration)
    if total_duration_bursts > duration:
        raise ValueError("Error! The total duration of bursts cannot exceed the total duration of the signal.")

    # Number of quiet periods (in between bursts)
    n_quiet = burst_number + 1  
    quiet_duration = np.repeat((duration - total_duration_bursts) / n_quiet, n_quiet)
    if burst_location is not None:
        if not (isinstance(burst_location, list) and all((isinstance(loc, int) or isinstance(loc, float)) for loc in burst_location)):
            raise TypeError("Error! 'burst_location' must be a list of integers or floats.")
        if len(burst_location) != burst_number:
            raise ValueError("Error! 'burst_location' list cannot be longer than the value of 'burst_number'.")
        if not all(np.round(((loc)/(1/sampling_rate)),10).is_integer() for loc in burst_location):
            warnings.warn("The precision derived from the selected sampling rate is not compatible with the burst locations. This might alter or compromise synthesization of the signals, increasing the sampling rate or altering the burst location/duration is recommended.")
        burst_location.sort()

        consecutive_bursts = False
        for i in range(len(burst_location)):
            loc = burst_location[i]
            dur = burst_duration[i]
            if loc < 0 or loc + dur > duration:
                raise ValueError("Error! Burst location times must be numbers greater than 0, and summed with their duration cannot be greater than the signal duration.")
            else:
                if i == 0: # First burst
                    quiet_duration[i] = loc
                    if burst_number == 1:
                        quiet_duration[i+1] = duration - (loc + dur)
                    else:
                        next_loc = burst_location[i+1]
                        if loc + dur >= next_loc: 
                            raise ValueError("Error! Burst location times overlap.")
                        elif loc + dur == next_loc - 1/sampling_rate:
                            consecutive_bursts = True
                elif i < burst_number - 1: # Intermediate burst
                    next_loc = burst_location[i+1]
                    if loc + dur >= next_loc: 
                            raise ValueError("Error! Burst location times overlap.")
                    elif loc + dur == next_loc - 1/sampling_rate:
                        consecutive_bursts = True
                    prev_loc = burst_location[i-1]
                    prev_dur = burst_duration[i-1]
                    quiet_duration[i] = loc - (prev_loc + prev_dur)
                else: # Final burst
                    prev_loc = burst_location[i-1]
                    prev_dur = burst_duration[i-1]
                    quiet_duration[i] = loc - (prev_loc + prev_dur)
                    quiet_duration[i+1] = duration - (loc + dur)
        if consecutive_bursts:
            warnings.warn("The defined burst locations and respective duration of each have lead to consecutive muscle activity bursts. This might hinder the distinction between bursts.")
    else:
        if not all(np.round((dur)/(1/sampling_rate),10).is_integer() for dur in quiet_duration):
            warnings.warn("The precision derived from the selected sampling rate is not compatible with the duration of quiet periods between evenly distributed bursts. This might alter (unequal duration of quiet periods) or compromise (impossible to separate bursts in the time domain) synthesization of the signals.")
    total_duration = total_duration_bursts + np.sum(quiet_duration)
    if np.round(total_duration,10) != duration:
        raise ValueError("Error! The total duration of bursts and quiet periods does not match the total duration of the signal.")
    
    # Generate muscle activity bursts
    bursts = []
    curr_samples = 0
    sigma_burst = 10 ** (SNR/20)
    for burst in range(burst_number):
        size = int(round(sampling_rate * burst_duration[burst]+1,0))
        signal = rng.normal(0, sigma_burst, size)
        truncated_gaussian = _truncated_gaussian_window(sigma,alpha,sampling_rate)
        bursts += [list(signal*truncated_gaussian + baseline)]
        curr_samples += size

    # Generate quiet periods
    quiets = []
    for quiet in range(n_quiet):  
        if quiet == 0:
            size = int(round(sampling_rate * quiet_duration[quiet],0))
        elif quiet != n_quiet - 1:
            size = int(round(sampling_rate * quiet_duration[quiet]-1,0))
        else:
            size = int(round(sampling_rate * quiet_duration[quiet]-1,0))
            # Guarantee signal length is respected (could be more or less samples due to rounding off errors, when using a sampling 
            # rate where the derived precision is not compatible with burst locations/duration or the duration of quiet periods):
            if length != curr_samples+size:
                size = length-curr_samples 
        curr_samples += size
        quiets += [list(np.zeros((size,)) + baseline)]

    # Merge muscle activity bursts and quiet periods
    emg = []
    activity = []
    for i in range(len(quiets)):  
        emg += quiets[i]
        activity += [list(np.zeros((len(quiets[i]),)))]
        if i < len(bursts):
            emg += bursts[i]
            activity += [list(np.ones((len(bursts[i]),)))]
    activity = np.concatenate(activity)
    emg = np.array(emg)

    # Add random (gaussian distributed) noise
    noise_signal = rng.normal(0, 1, length)
    emg_synth = emg + noise_signal

    t = np.arange(0,duration+1/sampling_rate,1/sampling_rate)

    params = {"duration": duration, 
              "length": length, 
              "sampling_rate": sampling_rate, 
              "SNR": SNR,
              "sigma": sigma,
              "alpha": alpha,
              "baseline": baseline,
              "burst_number": burst_number,
              "burst_location": burst_location,
              "random_state": rng,
              "clean_signal": emg,
              "noise_signal": noise_signal,
              "activity": activity}
    
    args = (emg_synth, t, params)
    names = ("emg", "t", "params")

    return utils.ReturnTuple(args, names)
