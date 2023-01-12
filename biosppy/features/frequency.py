# -*- coding: utf-8 -*-
"""
biosppy.features.frequency
--------------------------

This module provides methods to extract frequency features.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
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


def get_bands(frequencies, fband):
    band = np.argwhere((frequencies >= fband[0]) & (frequencies <= fband[1])).reshape(-1)

    return frequencies[band]


def freq_features(signal=None, sampling_rate=1000.):
    """Compute spectral metrics describing the signal.
   
    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    spectral_maxpeaks : int
        Number of peaks in the spectrum signal.
    spect_var : float
        Amount of the variation of the spectrum across time.
    curve_distance : float
        Euclidean distance between the cumulative sum of the signal spectrum and evenly spaced numbers across the signal lenght.
    spectral_roll_off : float
        Frequency so 95% of the signal energy is below that value.
    spectral_roll_on : float
        Frequency so 5% of the signal energy is below that value.
    spectral_dec : float
        Amount of decreasing in the spectral amplitude.
    spectral_slope : float
        Amount of decreasing in the spectral amplitude.
    spectral_centroid : float
        Centroid of the signal spectrum.
    spectral_spread : float
        Variance of the signal spectrum i.e. how it spreads around its mean value.
    spectral_kurtosis : float
        Kurtosis of the signal spectrum i.e. describes the flatness of the spectrum distribution.
    spectral_skewness : float
        Skewness of the signal spectrum i.e. describes the asymmetry of the spectrum distribution.
    max_frequency : float
        Maximum frequency of the signal spectrum maximum amplitude.
    fundamental_frequency : float
        Fundamental frequency of the signal.
    max_power_spectrum : float
        Spectrum maximum value.
    mean_power_spectrum : float
        Spectrum mean value.
    spectral_skewness : float
        Spectrum Skewness.
    spectral_kurtosis : float
        Spectrum Kurtosis.
    spectral_hist_{frequency band} : array
        Histogram of the signal spectrum on 0.05 - 0.1 (VLF), 0.1 - 0.2 (LF), 0.2 - 0.3 (MF), 0.3 - 0.4 (HF), 0.4 - 0.5 (VHF).

    References
    ----------
    - TSFEL library: https://github.com/fraunhoferportugal/tsfel
    - Peeters, Geoffroy. (2004). A large set of audio features for sound description (similarity and classification) in the CUIDADO project.
    - [0, 0.1], [0.1,0.2] , [0.2,0.3], [0.3, 0.4]: J. Wang and Y. Gong, “Recognition of multiple drivers’s emotional state,” in 2008 19th International Conference on Pattern Recognition, Dec 2008, pp. 1–4.
    - [0.05–5] was split into five bands - power + [0.05–1 Hz] - stat - Ghaderyan, P. and Abbasi, A., 2016. An efficient automatic workload estimation method based on electrodermal activity using pattern classifier combinations. International Journal of Psychophysiology, 110, pp.91-101.
    - temp fts on [0.05−0.50] was split into five bands + stats fts on FFT  - Shukla, Jainendra, et al. "Feature extraction and selection for emotion recognition from electrodermal activity." IEEE Transactions on Affective Computing 12.4 (2019): 857-869
    - FFT for bands (0.1, 0.2), F2 (0.2, 0.3) and F3 (0.3, 0.4) - Sánchez-Reolid, R., de la Rosa, F.L., Sánchez-Reolid, D., López, M.T., Fernández-Caballero, A. (2021). Feature and Time Series Extraction in Artificial Neural Networks for Arousal Detection from Electrodermal Activity. In: Rojas, I., Joya, G., Català, A. (eds) Advances in Computational Intelligence. IWANN 2021. Lecture Notes in Computer Science(), vol 12861. Springer, Cham. 

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    freqs, power = st.power_spectrum(signal, sampling_rate=sampling_rate, decibel=False)
    power = np.nan_to_num(power)

    args, names = [], []

    # temporal
    _fts = time.time_features(power, sampling_rate)
    fts_name = [str("FFT_" + i) for i in _fts.keys()]
    fts = list(_fts[:])

    args += fts
    names += fts_name

    # fundamental_frequency
    try:
        fundamental_frequency = freqs[np.argmax(power)]
    except Exception as e:
        print("fundamental frequency", e)
        fundamental_frequency = None
    args += [fundamental_frequency]
    names += ['fundamental_frequency']

    # harmonic sum
    try:
        if fundamental_frequency > (sampling_rate / 2 + 2):
            harmonics = np.array([n * fundamental_frequency for n in
                                  range(2, int((sampling_rate / 2) / fundamental_frequency), 1)]).astype(int)
            sp_hrm = power[np.array([np.where(freqs >= h)[0][0] for h in harmonics])]
            sum_harmonics = np.sum(sp_hrm)
        else:
            sum_harmonics = None
    except Exception as e:
        print("sum harmonics", e)
        sum_harmonics = None
    args += [sum_harmonics]
    names += ['sum_harmonics']

    # spectral_roll_on
    en_sp = power ** 2  # *(f[1]-f[0])
    cum_en = np.cumsum(en_sp)

    try:
        if cum_en[-1] is None or cum_en[-1] == 0.0:
            norm_cm_s = None
        else:
            norm_cm_s = cum_en / cum_en[-1]
    except Exception as e:
        print("norm_cm_s", e)
        norm_cm_s = None

    try:
        if norm_cm_s is None:
            spectral_roll_on = None
        else:
            spectral_roll_on = freqs[np.argwhere(norm_cm_s >= 0.05)[0][0]]
    except Exception as e:
        print("spectral_roll_on", e)
        spectral_roll_on = None

    args += [spectral_roll_on]
    names += ['spectral_roll_on']

    # spectral_roll_off
    try:
        if norm_cm_s is None:
            spectral_roll_off = None
        else:
            spectral_roll_off = freqs[np.argwhere(norm_cm_s >= 0.95)[0][0]]
    except Exception as e:
        print("spectral_roll_off", e)
        spectral_roll_off = None
    args += [spectral_roll_off]
    names += ['spectral_roll_off']

    # histogram
    try:
        _hist = list(np.histogram(power, bins=5)[0])
        _hist = _hist / np.sum(_hist)
    except Exception as e:
        print("frequency hist", e)
        _hist = [None] * 5

    args += [i for i in _hist]
    names += ['spectral_hist_' + str(i) for i in range(len(_hist))]

    # bands
    freqs, power = st.power_spectrum(signal, sampling_rate * 5, decibel=False)
    power = np.nan_to_num(power)

    # resampling
    _f = interpolate.interp1d(freqs, power)
    res_sr = 500  # new sampling rate
    freqs = np.arange(freqs[0], freqs[-1], 1 / res_sr)
    f_b = get_bands(freqs, fband=[0.05, 0.1])

    # temporal
    _fts = time.time_features(f_b, res_sr)
    fts_name = [str("FFT_VLF" + i) for i in _fts.keys()]
    fts = list(_fts[:])

    args += fts
    names += fts_name

    f_b = get_bands(freqs, fband=[0.1, 0.2])
    # temporal
    _fts = time.time_features(f_b, res_sr)
    fts_name = [str("FFT_LF" + i) for i in _fts.keys()]
    fts = list(_fts[:])

    args += fts
    names += fts_name

    f_b = get_bands(freqs, fband=[0.2, 0.3])
    # temporal
    _fts = time.time_features(f_b, res_sr)
    fts_name = [str("FFT_MD" + i) for i in _fts.keys()]
    fts = list(_fts[:])

    args += fts
    names += fts_name

    f_b = get_bands(freqs, fband=[0.3, 0.4])
    # temporal
    _fts = time.time_features(f_b, res_sr)
    fts_name = [str("FFT_HF" + i) for i in _fts.keys()]
    fts = list(_fts[:])

    args += fts
    names += fts_name

    f_b = get_bands(freqs, fband=[0.4, 0.5])
    # temporal
    _fts = time.time_features(f_b, res_sr)
    fts_name = [str("FFT_VHF" + i) for i in _fts.keys()]
    fts = list(_fts[:])

    args += fts
    names += fts_name

    # output
    args = tuple(args)
    names = tuple(names)

    return utils.ReturnTuple(args, names)
