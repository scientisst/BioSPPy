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

# local
from .. import utils
from ..signals import tools as st
from .. import stats


# variables
# Suggested frequency bands
EDA_FBANDS = {'VLF': (0.05, 0.1), 'LF': (0.1, 0.2), 'HF': (0.2, 0.3), 'VHF': (0.3, 0.4), 'UHF': (0.4, 0.5)}
EEG_FBANDS = {'Delta': (0.5, 4.), 'Theta': (4., 8.), 'Alpha': (8., 13.), 'Beta': (13., 30.), 'Gamma': (30., 100.)}
HRV_FBANDS = {'ULF': (0, 0.003), 'VLF': (0.003, 0.04), 'LF': (0.04, 0.15), 'HF': (0.15, 0.4), 'VHF': (0.4, 0.5)}


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

    Notes
    -----
    For the list of available features, check:
    - biosppy.signals.tools.signal_stats
    - biosppy.features.frequency.spectral_features
    - biosppy.features.frequency.compute_fbands

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

    # spectral features
    spectral_feats = spectral_features(freqs, power, sampling_rate)
    for arg, name in zip(spectral_feats, spectral_feats.keys()):
        feats = feats.append(arg, 'FFT_' + name)

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
    total_power : float
        Total power of the signal.
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
    out = out.append(total_power, 'total_power')

    # compute frequency bands
    for band_name, band_freq in fband.items():

        # check if the given frequency bands are within the range of the power spectrum
        if band_freq[0] < frequencies[0] or band_freq[1] > frequencies[-1]:
            out = out.append([None, None, None],
                             [band_name + '_power', band_name + '_rel_power', band_name + '_peak'])
            print("The frequency band '{}' is outside the range of the power spectrum.".format(band_name))
            continue

        # check if the lower bound is smaller than the upper bound
        if band_freq[0] > band_freq[1]:
            out = out.append([None, None, None],
                             [band_name + '_power', band_name + '_rel_power', band_name + '_peak'])
            print("The lower bound of the frequency band '{}' is larger than the upper bound.".format(band_name))
            continue

        # check if the frequency band difference is smaller than the frequency resolution
        if (band_freq[1] - band_freq[0]) < freq_res:
            out = out.append([None, None, None],
                             [band_name + '_power', band_name + '_rel_power', band_name + '_peak'])
            print("The frequency band '{}' is smaller than the frequency resolution.".format(band_name))
            continue

        band = np.where((frequencies >= band_freq[0]) & (frequencies <= band_freq[1]))[0]

        # compute band power
        band_power = np.sum(power[band]) * freq_res
        out = out.append(band_power, band_name + '_power')

        # compute relative power
        band_rel_power = band_power / total_power
        out = out.append(band_rel_power, band_name + '_rel_power')

        # compute peak frequency
        freq_peak = frequencies[np.argmax(power[band]) + band[0]]
        out = out.append(freq_peak, band_name + '_peak')

    return out


def spectral_features(freqs=None, power=None, sampling_rate=1000.):
    """Compute spectral features.

    Parameters
    ----------
    freqs : array
        Frequency values.
    power : array
        Power values.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    fundamental_frequency : float
        Fundamental frequency. The frequency with the highest power.
    sum_harmonics : float
        Sum of harmonics.
    roll_on : float
        Spectral roll on. The frequency where 95% of the total power is reached.
    roll_off : float
        Spectral roll off. The frequency where 5% of the total power is reached.
    centroid : float
        Spectral centroid. The weighted mean of the frequencies.
    slope : float
        Spectral slope. The slope of the linear regression of the power spectrum.
    spread : float
        Spectral spread. The standard deviation of the power spectrum.

    """

    # check inputs
    if any([power is None, freqs is None]):
        raise TypeError("Please specify all input parameters.")

    # ensure numpy
    power = np.array(power)
    freqs = np.array(freqs)

    # initialize output
    feats = utils.ReturnTuple((), ())

    # fundamental frequency
    fundamental_frequency = freqs[np.argmax(power)]
    feats = feats.append(fundamental_frequency, 'fundamental_frequency')

    # harmonic sum
    if fundamental_frequency > (sampling_rate / 2 + 2):
        harmonics = np.array([n * fundamental_frequency for n in
                              range(2, int((sampling_rate / 2) / fundamental_frequency), 1)]).astype(int)
        sp_hrm = power[np.array([np.where(freqs >= h)[0][0] for h in harmonics])]
        sum_harmonics = np.sum(sp_hrm)
    else:
        sum_harmonics = None
    feats = feats.append(sum_harmonics, 'sum_harmonics')

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
    feats = feats.append(spectral_roll_on, 'roll_on')

    # spectral roll off
    if norm_cm_s is None:
        spectral_roll_off = None
    else:
        spectral_roll_off = freqs[np.argwhere(norm_cm_s >= 0.95)[0][0]]
    feats = feats.append(spectral_roll_off, 'roll_off')

    # spectral centroid
    spectral_centroid = np.sum(power * freqs) / np.sum(power)
    feats = feats.append(spectral_centroid, 'centroid')

    # spectral slope
    spectral_slope = stats.linear_regression(freqs, power, show=False)['m']
    feats = feats.append(spectral_slope, 'slope')

    # spectral spread
    spectral_spread = np.sqrt(np.sum(power * (freqs - spectral_centroid) ** 2) / np.sum(power))
    feats = feats.append(spectral_spread, 'spread')

    return feats
