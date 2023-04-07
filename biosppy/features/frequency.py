# -*- coding: utf-8 -*-
"""
biosppy.features.frequency
--------------------------

This module provides methods to extract frequency features.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# 3rd party
import numpy as np
from scipy import interpolate

# local
from .. import utils
from . import time
from ..signals import tools as st
from .. import stats


def frequency(signal=None, sampling_rate=1000., fbands=None):
    """Compute spectral metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    fbands : dict
        Frequency bands to compute the features, where the keys are the names of the bands and the values are the
        frequency ranges (in Hz) of the bands.

    Returns
    -------
    feats : ReturnTuple object
        Frequency features of the signal.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # initialize output
    feats = utils.ReturnTuple((), ())

    # Compute power spectrum
    freqs, power = st.power_spectrum(signal, sampling_rate=sampling_rate, decibel=False)

    # basic stats
    signal_feats = st.signal_stats(power)
    for arg, name in zip(signal_feats, signal_feats.keys()):
        feats = feats.append(arg, 'FFT_' + name)

    # fundamental frequency
    fundamental_frequency = freqs[np.argmax(power)]
    feats = feats.append(fundamental_frequency, 'FFT_fundamental_frequency')

    # harmonic sum
    if fundamental_frequency > (sampling_rate / 2 + 2):
        harmonics = np.array([n * fundamental_frequency for n in
                              range(2, int((sampling_rate / 2) / fundamental_frequency), 1)]).astype(int)
        sp_hrm = power[np.array([np.where(freqs >= h)[0][0] for h in harmonics])]
        sum_harmonics = np.sum(sp_hrm)
    else:
        sum_harmonics = None
    feats = feats.append(sum_harmonics, 'FFT_sum_harmonics')

    # spectral roll on
    en_sp = power ** 2
    cum_en = np.cumsum(en_sp)

    if cum_en[-1] is None or cum_en[-1] == 0.0:
        norm_cm_s = None
    else:
        norm_cm_s = cum_en / cum_en[-1]

    if norm_cm_s is not None:
        spectral_roll_on = freqs[np.argwhere(norm_cm_s >= 0.05)[0][0]]
    else:
        spectral_roll_on = None
    feats = feats.append(spectral_roll_on, 'FFT_spectral_roll_on')

    # spectral roll off
    if norm_cm_s is None:
        spectral_roll_off = None
    else:
        spectral_roll_off = freqs[np.argwhere(norm_cm_s >= 0.95)[0][0]]
    feats = feats.append(spectral_roll_off, 'FFT_spectral_roll_off')

    # spectral centroid
    spectral_centroid = np.sum(power * freqs) / np.sum(power)
    feats = feats.append(spectral_centroid, 'FFT_spectral_centroid')

    # spectral slope
    spectral_slope = stats.linear_regression(freqs, power, show=False)['m']
    feats = feats.append(spectral_slope, 'FFT_spectral_slope')

    # spectral spread
    spectral_spread = np.sqrt(np.sum(power * (freqs - spectral_centroid) ** 2) / np.sum(power))
    feats = feats.append(spectral_spread, 'FFT_spectral_spread')

    # histogram
    fft_hist = stats.histogram(power, bins=5, normalize=True)
    for arg, name in zip(fft_hist, fft_hist.keys()):
        feats = feats.append(arg, 'FFT_' + name)

    # frequency bands
    if fbands is not None:
        fband_feats = compute_fbands(freqs, power, fbands)
        feats = feats.join(fband_feats)

    return feats


def compute_fbands(frequencies=None, power=None, fband=None):
    """Compute frequency bands.

    Parameters
    ----------
    frequencies : array
        Frequency values.
    power : array
        Power values.
    fband : dict
        Frequency bands to compute the features, where the keys are the names of the bands and the values are
        two-element lists/tuples with the lower and upper frequency bounds (in Hz) of the bands.

    Returns
    -------
    {fband}_power : float
        Power of the frequency band.
    {fband}_rel_power : float
        Relative power of the frequency band.
    {fband}_peak : float
        Peak frequency of the frequency band.

    """

    # check inputs
    if any([frequencies is None, power is None, fband is None]):
        raise TypeError("Please specify all input parameters.")

    # ensure numpy
    frequencies = np.array(frequencies)
    power = np.array(power)

    # initialize output
    out = utils.ReturnTuple((), ())

    # frequency resolution
    freq_res = frequencies[1] - frequencies[0]

    # total power
    total_power = np.sum(power) * freq_res

    # compute frequency bands
    for band_name, band_freq in fband.items():
        band = np.where((frequencies >= band_freq[0]) & (frequencies <= band_freq[1]))[0]

        # compute band power
        band_power = np.sum(power[band]) * freq_res
        out = out.append(band_power, band_name + '_power')

        # compute relative power
        band_rel_power = band_power / total_power
        out = out.append(band_rel_power, band_name + '_rel_power')

        # compute peak frequency
        freq_peak = frequencies[np.argmax(power[band])]
        out = out.append(freq_peak, band_name + '_peak')

    return out
