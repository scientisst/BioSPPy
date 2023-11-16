# -*- coding: utf-8 -*-
"""
biosppy.features.cepstral
-------------------------

This module provides methods to extract cepstral features.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# 3rd party
import numpy as np
from scipy import fft

# local
from .. import utils
from . import time
from ..signals import tools as st


def cepstral(signal=None, sampling_rate=1000.):
    """Compute quefrency metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    feats : ReturnTuple object
        Time features computed over the signal mel-frequency cepstral coefficients.

    Notes
    -----
    Check biosppy.features.time for the list of time features.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # initialize output
    feats = utils.ReturnTuple((), ())

    # compute mel coefficients
    mel_coeff = mfcc(signal, sampling_rate)["mfcc"]

    # time features
    time_feats = time.time(signal=mel_coeff, sampling_rate=sampling_rate)
    for arg, name in zip(time_feats, time_feats.keys()):
        feats = feats.append(arg, 'mfcc_' + name)

    return feats


def freq_to_mel(hertz):
    """Converts mel-frequencies to hertz frequencies.
    
    Parameters
    ----------
    hertz : array
        Hertz frequencies.
 
    Returns
    -------
    mel frequencies : array
        Mel frequencies.
    
    References
    ----------
    .. [Kool12] Shashidhar G. Koolagudi, Deepika Rastogi, K. Sreenivasa Rao, Identification of Language using
    Mel-Frequency Cepstral Coefficients (MFCC), Procedia Engineering, Volume 38, 2012, Pages 3391-3398, ISSN 1877-7058
    
    """   

    return 1125 * np.log(1 + hertz / 700)


def mel_to_freq(mel):
    """Converts mel-frequencies to hertz frequencies.
    
    Parameters
    ----------
    mel : array
        Mel frequencies.
 
    Returns
    -------
    hertz frequencies : array
        Hertz frequencies.

    References
    ----------
    .. [Kool12] Shashidhar G. Koolagudi, Deepika Rastogi, K. Sreenivasa Rao, Identification of Language using
    Mel-Frequency Cepstral Coefficients (MFCC), Procedia Engineering, Volume 38, 2012, Pages 3391-3398, ISSN 1877-7058

    """

    return 700 * (np.exp(mel / 1125) - 1)


def mfcc(signal=None, sampling_rate=1000., window_size=100, num_filters=10):
    """Computes the mel-frequency cepstral coefficients.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    window_size : int
       DFT window size. 
    num_filters : int
       Number of filters.

    Returns
    -------
    mfcc : array
        Signal mel-frequency cepstral coefficients.

    References
    ----------
    .. [Haytham16] Fayek, Haytham. "Speech Processing for Machine Learning: Filter
    banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between."
    Blog post. 2016. https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    .. 'Vanilla STFT and MFCC' by brihijoshi, accessed in october 2022:
    https://github.com/brihijoshi/vanilla-stft-mfcc/
    .. 'Time Series Feature Extraction Library' by fraunhoferportugal, accessed in october 2022:
    https://github.com/fraunhoferportugal/tsfel/
    .. 'MFCC implementation and tutorial' by ilyamich, accessed in october 2022:
    https://www.kaggle.com/code/ilyamich/mfcc-implementation-and-tutorial

   """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # compute power spectrum
    freqs, power = st.power_spectrum(signal, sampling_rate=sampling_rate, decibel=False)

    # filter bank
    low_f = 0
    high_f = freqs[-1]

    # convert to mel
    low_f_mel = freq_to_mel(low_f)
    high_f_mel = freq_to_mel(high_f)

    # linearly spaced array between the two MEL frequencies
    lin_mel = np.linspace(low_f_mel, high_f_mel, num=num_filters+2)
    
    # convert the array to the frequency space
    lin_hz = np.array([mel_to_freq(d) for d in lin_mel])
    
    # normalize the array to the FFT size and choose the associated FFT values
    filter_bins_hz = np.floor((window_size + 1) / sampling_rate * lin_hz).astype(int)
    
    # filter bank
    filter_banks = []

    # iterate bins
    for b in range(len(filter_bins_hz)-2):
        _f = [0]*(filter_bins_hz[b])
        _f += np.linspace(0, 1, filter_bins_hz[b + 1] - filter_bins_hz[b]).tolist()
        _f += np.linspace(1, 0, filter_bins_hz[b + 2] - filter_bins_hz[b + 1]).tolist()
        pad = len(freqs) - filter_bins_hz[b + 2]
        if pad > 0:
            _f += [0]*pad
        else:
            _f = _f[:len(freqs)]

        filter_banks += [np.array(_f)]
    filter_banks = np.array(filter_banks)
    
    enorm = 2.0 / (lin_hz[2:num_filters+2] - lin_hz[:num_filters])
    filter_banks *= enorm[:, np.newaxis]

    signal_power = np.abs(power)**2*(1/len(power))
    
    filter_banks = np.dot(filter_banks, signal_power.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    mel_coeff = fft.dct(filter_banks)[1:]  # Keep 2-13

    mel_coeff -= (np.mean(mel_coeff, axis=0) + 1e-8)  # norm

    # sinusoidal liftering to the MFCCs to de-emphasize higher MFCCs
    n = np.arange(len(mel_coeff))
    cep_lifter = 22
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mel_coeff *= lift  

    # output
    args = (mel_coeff,)
    names = ("mfcc",)

    return utils.ReturnTuple(args, names)
