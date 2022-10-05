# -*- coding: utf-8 -*-
"""
biosppy.signals.hrv
-------------------

This module provides computation and visualization of Heart-Rate Variability
metrics.


:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
# compat
from __future__ import absolute_import, division, print_function

# 3rd party
import numpy as np
import warnings
from scipy.interpolate import interp1d
from scipy.signal import welch

# local
from .. import utils
from .. import plotting
from . import tools

# Global variables
FBANDS = {'ulf': [0, 0.003],
          'vlf': [0.003, 0.04],
          'lf': [0.04, 0.15],
          'hf': [0.15, 0.4],
          'vhf': [0.4, 0.5]
          }

SPECTRUM_COLORS = {'ulf': '#e6eff6',
                   'vlf': '#89b4c4',
                   'lf': '#548999',
                   'hf': '#f1d3a1',
                   'vhf': '#e3dbd9'
                   }


def compute_rri(rpeaks, sampling_rate=1000.):
    """ Computes RR intervals in milliseconds from a list of R-peak indexes.

    Parameters
    ----------
    rpeaks : list, array
        R-peak index locations.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    rri : array
        RR-intervals (ms).
    """

    # ensure input format
    rpeaks = np.array(rpeaks)

    # difference of R-peaks converted to ms
    rri = (1000. * np.diff(rpeaks)) / sampling_rate

    # check if rri is within physiological parameters
    if rri.min() < 400 or rri.min() > 1400:
        warnings.warn("RR-intervals appear to be out of normal parameters. Check input values.")

    return rri


def filter_rri(rri=None, threshold=1200):
    """Filters an RRI sequence based on a maximum threshold in milliseconds.

    Parameters
    ----------
    rri : array
        RR-intervals (default: ms).
    threshold : int, float, optional
        Maximum rri value to accept (ms).
    """

    # ensure input format
    rri = np.array(rri, dtype=float)

    # filter rri values
    rri_filt = rri[np.where(rri < threshold)]

    return rri_filt


def hrv_timedomain(rri, duration=None, show=False, detrend=True):
    """ Computes the time domain HRV features from a sequence of RR intervals in milliseconds.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    show : bool, optional
        Controls the plotting calls.
    detrend : bool, optional
        Whether to detrend the input signal.

    Returns
    -------
    hr : array
        Heart rate (bpm).
    hr_min : float
        Minimum heart rate (bpm).
    hr_max : float
        Maximum heart rate (bpm).
    hr_minmax :  float
        Difference between the highest and the lowest heart rate (bpm).
    hr_avg : float
        Average heart rate (bpm).
    rr_mean : float
        Mean value of RR intervals (ms).
    rmssd : float
        RMSSD - Root mean square of successive RR interval differences (ms).
    nn50 : int
        NN50 - Number of successive RR intervals that differ by more than 50ms.
    pnn50 : float
        pNN50 - Percentage of successive RR intervals that differ by more than
        50ms.
    sdnn: float
       SDNN - Standard deviation of RR intervals (ms).
    """

    # check inputs
    if rri is None:
        raise ValueError("Please specify an RRI list or array.")

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # detrend
    if detrend:
        rri_det = detrend_window(rri)
    else:
        rri_det = rri

    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 10:
        raise ValueError("Signal duration must be greater than 10 seconds to compute time-domain features.")

    # initialize outputs
    out = utils.ReturnTuple((), ())

    # compute the difference between RRIs
    rri_diff = np.diff(rri_det)

    if duration >= 10:
        # compute heart rate features
        hr = 60/(rri/1000)  # bpm
        hr_min = hr.min()
        hr_max = hr.max()
        hr_minmax = hr.max() - hr.min()
        hr_avg = hr.mean()

        out = out.append([hr, hr_min, hr_max, hr_minmax, hr_avg],
                         ['hr', 'hr_min', 'hr_max', 'hr_minmax', 'hr_avg'])

        # compute RRI features
        rr_mean = rri.mean()
        rmssd = (rri_diff ** 2).mean() ** 0.5

        out = out.append([rr_mean, rmssd], ['rr_mean', 'rmssd'])

    if duration >= 20:
        # compute NN50 and pNN50
        th50 = 50
        nntot = len(rri_diff)
        nn50 = len(np.argwhere(abs(rri_diff) > th50))
        pnn50 = 100 * (nn50 / nntot)

        out = out.append([nn50, pnn50], ['nn50', 'pnn50'])

    if duration >= 60:
        # compute SDNN
        sdnn = rri_det.std()

        out = out.append(sdnn, 'sdnn')

    if duration >= 90:
        # compute geometrical features (histogram)
        hti, tinn = compute_geometrical(rri=rri, show=show)

        out = out.append([hti, tinn], ['hti', 'tinn'])

    return out


def hrv_frequencydomain(rri=None, duration=None, freq_method='FFT', fbands=None, detrend=True):
    """Computes the frequency domain HRV features from a sequence of RR intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    freq_method : str, optional
        Method for spectral estimation. If 'FFT' uses Welch's method.
    fbands : dict, optional
        Dictionary specifying the desired HRV frequency bands.
    detrend : bool, optional
        Whether to detrend the input signal. Default: True.

    Returns
    -------
    vlf_peak : float
        Peak frequency (Hz) of the very-low-frequency band (0.0033–0.04 Hz) in
        normal units.
    vlf_pwr : float
        Relative power of the very-low-frequency band (0.0033–0.04 Hz) in
        normal units.
    lf_peak : float
        Peak frequency (Hz) of the low-frequency band (0.04–0.15 Hz).
    lf_pwr : float
        Relative power of the low-frequency band (0.04–0.15 Hz) in normal
        units.
    hf_peak : float
        Peak frequency (Hz)  of the high-frequency band (0.15–0.4 Hz).
    hf_pwr : float
        Relative power of the high-frequency band (0.15–0.4 Hz) in normal
        units.
    lf_hf : float
        Ratio of LF-to-HF power.
    total_pwr : float
        Total power.
    """

    # check inputs
    if rri is None:
        raise ValueError("Please specify an RRI list or array.")

    freq_methods = ['FFT']
    if freq_method not in freq_methods:
        raise ValueError(f"'{freq_method}' is not an available input. Choose one from: {freq_methods}.")

    if fbands is None:
        fbands = FBANDS

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # ensure minimal duration
    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 20:
        raise IOError("Signal duration must be greater than 20 seconds to compute frequency-domain features.")

    # initialize outputs
    out = utils.ReturnTuple((), ())

    # resampling with cubic interpolation for equidistant samples
    fs = 1
    t = np.cumsum(rri)
    t -= t[0]
    f_inter = interp1d(t, rri, 'cubic')
    t_inter = np.arange(t[0], t[-1], 1000. / fs)
    rri_inter = f_inter(t_inter)

    # detrend
    if detrend:
        rri_inter = detrend_window(rri_inter)

    if duration >= 20:

        # compute frequencies and powers
        if freq_method == 'FFT':
            frequencies, powers = welch(rri_inter, fs=fs, scaling='density', nperseg=300)

        # compute frequency bands
        fb_out = compute_fbands(frequencies=frequencies, powers=powers)

        out = out.join(fb_out)

        # compute LF/HF ratio
        lf_hf = fb_out['lf_pwr'] / fb_out['hf_pwr']

        out = out.append(lf_hf, 'lf_hf')

    return out


def hrv_nonlinear(rri=None, duration=None, detrend=True):
    """ Computes the non-linear HRV features from a sequence of RR intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    detrend : bool, optional
        Whether to detrend the input signal. Default: True.

    Returns
    -------
    s : float
        S - Area of the ellipse of the Poincaré plot (ms^2).
    sd1 : float
        SD1 - Poincaré plot standard deviation perpendicular to the identity
        line (ms).
    sd2 : float
        SD2 - Poincaré plot standard deviation along the identity line (ms).
    sd12 : float
        SD1/SD2 - SD1 to SD2 ratio.
    """

    # check inputs
    if rri is None:
        raise TypeError("Please specify an RRI list or array.")

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # check duration
    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 90:
        raise IOError("Signal duration must be greater than 90 seconds to compute non-linear features.")

    # detrend
    if detrend:
        rri = detrend_window(rri)

    # initialize outputs
    out = utils.ReturnTuple((), ())

    if duration >= 90:
        # compute SD1, SD2, SD1/SD2 and S
        cp = compute_poincare(rri=rri)

        out = out.join(cp)

    return out


def compute_fbands(frequencies, powers, fbands=None):
    """ Computes frequency domain features for the specified frequency bands.

    Parameters
    ----------
    frequencies : array
        Frequency axis.
    powers : array
        Power spectrum values for the frequency axis-
    fbands : dict, optional
        Dictionary containing the limits of the frequency bands.

    Returns
    -------
    _peak : float
        Peak frequency of the frequency band (Hz).
    _pwr : float
        Absolute power of the frequency band (ms^2).
    _rpwr : float
        Relative power of the frequency band (nu).
    """

    # initialize outputs
    out = utils.ReturnTuple((), ())

    df = frequencies[1] - frequencies[0]  # frequency resolution
    total_pwr = np.sum(powers) * df

    if fbands is None:
        fbands = FBANDS

    # compute power, peak and relative power for each frequency band
    for fband in fbands.keys():
        band = np.argwhere((frequencies >= fbands[fband][0]) & (frequencies <= fbands[fband][-1])).reshape(-1)

        # check if it's possible to compute the frequency band
        if len(band) == 0:
            continue

        pwr = np.sum(powers[band]) * df
        peak = frequencies[band][np.argmax(powers[band])]
        rpwr = pwr / total_pwr

        out = out.append([pwr, peak, rpwr], [fband + '_pwr', fband + '_peak', fband + '_rpwr'])

    return out


def compute_poincare(rri, show=False):
    """ Compute the Poincaré features from a sequence of RR intervals.

    Parameters
    ----------
     rri : array
        RR-intervals (ms).
    show : bool, optional
        If True, show a the Poincaré plot.

    Returns
    -------
    s : float
        S - Area of the ellipse of the Poincaré plot (ms^2).
    sd1 : float
        SD1 - Poincaré plot standard deviation perpendicular to the identity
        line (ms).
    sd2 : float
        SD2 - Poincaré plot standard deviation along the identity line (ms).
    sd12 : float
        SD1/SD2 - SD1 to SD2 ratio.
    """

    # initialize outputs
    out = utils.ReturnTuple((), ())

    x = rri[:-1]
    y = rri[1:]

    # compute SD1, SD2 and S
    x1 = (x - y) / np.sqrt(2)
    x2 = (x + y) / np.sqrt(2)
    sd1 = x1.std()
    sd2 = x2.std()
    s = np.pi * sd1 * sd2

    # compute sd1/sd2 ratio
    sd12 = sd1 / sd2

    # output
    out = out.append([s, sd1, sd2, sd12], ['s', 'sd1', 'sd2', 'sd12'])

    if show:
        plotting.plot_poincare(rri, x=x, y=y, s=s, sd1=sd1, sd2=sd2, sd12=sd12)

    return out


def compute_geometrical(rri, binsize=1/128, show=False, detailed=False):
    """ Computes the geometrical features from a sequence of RR intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    binsize : float, optional
        Binsize for RRI histogram (s). Default: 1/128 s.
    show : bool, optional
        If True, show the RRI histogram. Default: False.
    detailed : bool, optional
        If True, returns the histogram variables (for plotting purposes).

    Returns
    -------
    hti : float
        HTI - HRV triangular index - Integral of the density of the RR interval
        histogram divided by its height.
    tinn : float
        TINN - Baseline width of RR interval histogram (ms).

    """
    binsize = binsize * 1000  # to ms

    # create histogram
    tmin = rri.min()
    tmax = rri.max()
    bins = np.arange(tmin, tmax + binsize, binsize)
    nn_hist = np.histogram(rri, bins)

    # histogram peak
    max_count = np.max(nn_hist[0])
    peak_hist = np.argmax(nn_hist[0])

    # compute HTI
    hti = len(rri) / max_count

    # possible N and M values
    n_values = bins[:peak_hist]
    m_values = bins[peak_hist + 1:]

    # find triangle with base N and M that best approximates the distribution
    error_min = np.inf
    n = 0
    m = 0
    q_hist = None

    for n_ in n_values:

        for m_ in m_values:

            t = np.array([tmin, n_, nn_hist[1][peak_hist], m_, tmax + binsize])
            y = np.array([0, 0, max_count, 0, 0])
            q = interp1d(x=t, y=y, kind='linear')
            q = q(bins)

            # compute the sum of squared differences
            error = np.sum((nn_hist[0] - q[:-1]) ** 2)

            if error < error_min:
                error_min = error
                n, m, q_hist = n_, m_, q

    # compute TINN
    tinn = m - n

    # plot
    if show:
        plotting.plot_hrv_hist(rri=rri,
                  bins=bins,
                  hist=q_hist,
                  hti=hti,
                  tinn=tinn)

    # output
    out = utils.ReturnTuple([hti, tinn], ['hti', 'tinn'])

    if detailed:
        out = utils.ReturnTuple([bins, q_hist, hti, tinn], ['bins', 'hist', 'hti', 'tinn'])

    return out


def detrend_window(rri, win_len=2000, **kwargs):
    """ Facilitates RRI detrending method using a signal window.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    win_len : int, optional
        Length of the window to detrend the RRI signal. Default: 2000.
    kwargs : dict, optional
        Parameters of the detrending method.
    Returns
    -------
    rri_det : array
        Detrended RRI signal.

    """

    # check input type
    win_len = int(win_len)

    # extract parameters
    if kwargs is None:
        smoothing_factor = 500
    else:
        smoothing_factor = kwargs['smoothing_factor']

    # detrend signal
    if len(rri) > win_len:
        # split the signal
        splits = int(len(rri)/win_len)
        rri_splits = np.array_split(rri, splits)

        # compute the detrended signal for each split
        rri_det = []
        for split in rri_splits:
            split_det = tools.detrend_smoothness_priors(split, smoothing_factor)['detrended']
            rri_det.append(split_det)

        # concantenate detrended splits
        rri_det = np.concatenate(rri_det)

    else:
        rri_det = tools.detrend_smoothness_priors(rri, smoothing_factor)['detrended']

    return rri_det
