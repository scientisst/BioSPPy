# -*- coding: utf-8 -*-
"""
biosppy.signals.hrv
-------------------

This module provides computation and visualization of Heart-Rate Variability
metrics.


:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
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
from . import tools as st

# Global variables
FBANDS = {'ulf': [0, 0.003],
          'vlf': [0.003, 0.04],
          'lf': [0.04, 0.15],
          'hf': [0.15, 0.4],
          'vhf': [0.4, 0.5]
          }

NOT_FEATURES = ['rri', 'rri_trend', 'outliers_method', 'rri_det', 'hr', 'bins',
                'q_hist', 'fbands', 'frequencies', 'powers', 'freq_method']


def hrv(rpeaks=None, sampling_rate=1000., rri=None, parameters='auto',
        outliers='interpolate', detrend_rri=True, features_only=True,
        show=True, show_individual=False, **kwargs):
    """Extracts the RR-interval sequence from a list of R-peak indexes and
    extracts HRV features.

    Parameters
    ----------
    rpeaks : array
        R-peak index locations.
    sampling_rate : int, float, optional
        Sampling frequency (Hz). Default: 1000.0 Hz.
    rri : array, optional
        RR-intervals (ms). Providing this parameter overrides the computation of
        RR-intervals from rpeaks.
    parameters : str, optional
        If 'auto' computes the recommended HRV features. If 'time' computes
        only time-domain features. If 'frequency' computes only
        frequency-domain features. If 'non-linear' computes only non-linear
        features. If 'all' computes all available HRV features. Default: 'auto'.
    outliers : str, optional
        Determines the method to handle outliers. If 'interpolate', replaces
        the outlier RR-intervals
        with cubic spline interpolation based on a local threshold. If 'filter',
        the RR-interval sequence
        is cut at the outliers. If None, no correction is performed. Default:
        'interpolate'.
    detrend_rri : bool, optional
        Whether to detrend the RRI sequence with the default method smoothness
        priors. Default: True.
    features_only : bool, optional
        Whether to return only the hrv features. Default: True.
    show : bool, optional
        Whether to show the HRV summary plot. Default: True.
    show_individual : bool, optional
        Whether to show the individual HRV plots. Default: False.
    kwargs : dict, optional
        fbands : dictionary of frequency bands (Hz) to use.

    Returns
    -------
    rri : array
        RR-intervals (ms).
    rri_det : array
        Detrended RR-interval sequence (ms), if detrending was applied.
    hrv_features : dict
        The set of HRV features extracted from the RRI data. The number of
        features depends on the chosen parameters.
    """

    # check inputs
    if rpeaks is None and rri is None:
        raise TypeError("Please specify an R-Peak or RRI list or array.")

    parameters_list = ['auto', 'time', 'frequency', 'non-linear', 'all']
    if parameters not in parameters_list:
        raise ValueError(f"'{parameters}' is not an available input. Enter one"
                         f"from: {parameters_list}.")

    # ensure input format
    sampling_rate = float(sampling_rate)

    # initialize outputs
    out = utils.ReturnTuple((), ())
    hrv_td, hrv_fd, hrv_nl = None, None, None

    # compute RRIs
    if rri is None:
        rpeaks = np.array(rpeaks, dtype=float)
        rri = compute_rri(rpeaks=rpeaks, sampling_rate=sampling_rate,
                          filter_rri=False)

    # compute duration
    duration = np.sum(rri) / 1000.  # seconds

    # handle outliers
    if outliers is None:
        pass
    elif outliers == 'interpolate':
        rri = rri_correction(rri)
    elif outliers == 'filter':
        rri = rri_filter(rri)

    # add rri to output
    out = out.append([rri, str(outliers)], ['rri', 'outliers_method'])

    # detrend rri sequence
    if detrend_rri:
        rri_det, rri_trend = detrend_window(rri)
        # add to output
        out = out.append([rri_det, rri_trend], ['rri_det', 'rri_trend'])
    else:
        rri_det = None
        rri_trend = None

    # plot
    if show_individual:
        plotting.plot_rri(rri, rri_trend, show=show_individual)

    # extract features
    if parameters == 'all':
        duration = np.inf

    # compute time-domain features
    if parameters in ['time', 'auto', 'all']:
        try:
            hrv_td = hrv_timedomain(rri=rri,
                                    duration=duration,
                                    detrend_rri=detrend_rri,
                                    show=show_individual,
                                    rri_detrended=rri_det)
            out = out.join(hrv_td)

        except ValueError as e:
            print('WARNING: Time-domain features not computed. Check input.')
            print(e)

            pass

    # compute frequency-domain features
    if parameters in ['frequency', 'auto', 'all']:
        try:
            hrv_fd = hrv_frequencydomain(rri=rri,
                                         duration=duration,
                                         detrend_rri=detrend_rri,
                                         show=show_individual,
                                         fbands=kwargs.get('fbands', None))
            out = out.join(hrv_fd)
        except ValueError as e:
            print('WARNING: Frequency-domain features not computed. Check input.')
            print(e)
            pass

    # compute non-linear features
    if parameters in ['non-linear', 'auto', 'all']:
        try:
            hrv_nl = hrv_nonlinear(rri=rri,
                                   duration=duration,
                                   detrend_rri=detrend_rri,
                                   show=show_individual)
            out = out.join(hrv_nl)
                
        except ValueError as e:
            print('WARNING: Non-linear features not computed. Check input.')
            print(e)
            pass

    # plot summary
    if show:
        if hrv_td is not None and hrv_fd is not None and hrv_nl is not None:
            plotting.plot_hrv(rri=rri,
                              rri_trend=rri_trend,
                              td_out=hrv_td,
                              nl_out=hrv_nl,
                              fd_out=hrv_fd,
                              show=True,
                              )
        else:
            warning = "Not all features were computed. To show the summary " \
                      "plot all features must be computed. Set " \
                      "'show_individual' to True to show the individual " \
                      "plots, or use parameters='all' to compute all features."
            warnings.warn(warning)

    # clean output if features_only
    if features_only:
        for key in NOT_FEATURES:
            try:
                out = out.delete(key)
            except ValueError:
                pass

    return out


def compute_rri(rpeaks, sampling_rate=1000., filter_rri=True, show=False):
    """Computes RR intervals in milliseconds from a list of R-peak indexes.

    Parameters
    ----------
    rpeaks : list, array
        R-peak index locations.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    filter_rri : bool, optional
        Whether to filter the RR-interval sequence. Default: True.
    show : bool, optional
        Plots the RR-interval sequence. Default: False.

    Returns
    -------
    rri : array
        RR-intervals (ms).
    """

    # ensure input format
    rpeaks = np.array(rpeaks)

    # difference of R-peaks converted to ms
    rri = (1000. * np.diff(rpeaks)) / sampling_rate

    # filter rri sequence
    if filter_rri:
        rri = rri_filter(rri)

    # check if rri is within physiological parameters
    if rri.min() < 400 or rri.min() > 1400:
        warnings.warn("RR-intervals appear to be out of normal parameters."
                      "Check input values.")

    if show:
        plotting.plot_rri(rri)

    return rri


def rri_filter(rri=None, threshold=1200):
    """Filters an RRI sequence based on a maximum threshold in milliseconds.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    threshold : int, float, optional
        Maximum rri value to accept (ms).

    Returns
    -------
    rri_filt : array
        Filtered RR-intervals (ms).
    """

    # ensure input format
    rri = np.array(rri, dtype=float)

    # filter rri values
    rri_filt = rri[np.where(rri < threshold)]

    return rri_filt


def rri_correction(rri=None, threshold=250):
    """Corrects artifacts in an RRI sequence based on a local average threshold.
    Artifacts are replaced with cubic spline interpolation.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    threshold : int, float, optional
        Local average threshold (ms). Default: 250.

    Returns
    -------
    rri : array
        Corrected RR-intervals (ms).
    """

    # check inputs
    if rri is None:
        raise ValueError("Please specify an RRI list or array.")

    # ensure input format
    rri = np.array(rri, dtype=float)

    # compute local average
    rri_filt, _ = st.smoother(signal=rri, kernel='median', size=3)

    # find artifacts
    artifacts = np.abs(rri - rri_filt) > threshold

    # replace artifacts with cubic spline interpolation
    rri[artifacts] = interp1d(np.where(~artifacts)[0], rri[~artifacts],
                              kind='cubic')(np.where(artifacts)[0])

    return rri


def hrv_timedomain(rri, duration=None, detrend_rri=True, show=False, **kwargs):
    """Computes the time domain HRV features from a sequence of RR intervals
    in milliseconds.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    detrend_rri : bool, optional
        Whether to detrend the input signal.
    show : bool, optional
        Controls the plotting calls. Default: False.

    Returns
    -------
    hr : array
        Instantaneous heart rate (bpm).
    hr_min : float
        Minimum heart rate (bpm).
    hr_max : float
        Maximum heart rate (bpm).
    hr_minmax :  float
        Difference between the highest and the lowest heart rates (bpm).
    hr_mean : float
        Mean heart rate (bpm).
    hr_median : float
        Median heart rate (bpm).
    rr_min : float
        Minimum value of RR intervals (ms).
    rr_max : float
        Maximum value of RR intervals (ms).
    rr_minmax :  float
        Difference between the highest and the lowest values of RR intervals (ms).
    rr_mean : float
        Mean value of RR intervals (ms).
    rr_median : float
        Median value of RR intervals (ms).
    rmssd : float
        RMSSD - Root mean square of successive RR interval differences (ms).
    nn50 : int
        NN50 - Number of successive RR intervals that differ by more than 50ms.
    pnn50 : float
        pNN50 - Percentage of successive RR intervals that differ by more than
        50ms.
    sdnn: float
       SDNN - Standard deviation of RR intervals (ms).
    hti : float
        HTI - HRV triangular index - Integral of the density of the RR interval
        histogram divided by its height.
    tinn : float
        TINN - Baseline width of RR interval histogram (ms).
    """

    # check inputs
    if rri is None:
        raise ValueError("Please specify an RRI list or array.")

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # detrend
    if detrend_rri:
        if 'rri_detrended' in kwargs:
            rri_det = kwargs['rri_detrended']
        else:
            rri_det = detrend_window(rri)['rri_det']
        print('Time domain: the rri sequence was detrended.')
    else:
        rri_det = rri

    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 10:
        raise ValueError("Signal duration must be greater than 10 seconds to "
                         "compute time-domain features.")

    # initialize outputs
    out = utils.ReturnTuple((), ())

    # compute the difference between RRIs
    rri_diff = np.diff(rri_det)

    if duration >= 10:
        # compute heart rate features
        hr = 60 / (rri / 1000)  # bpm
        hr_min = hr.min()
        hr_max = hr.max()
        hr_minmax = hr.max() - hr.min()
        hr_mean = hr.mean()
        hr_median = np.median(hr)

        out = out.append([hr, hr_min, hr_max, hr_minmax, hr_mean, hr_median],
                         ['hr', 'hr_min', 'hr_max', 'hr_minmax', 'hr_mean',
                          'hr_median'])

        # compute RRI features
        rr_min = rri.min()
        rr_max = rri.max()
        rr_minmax = rri.max() - rri.min()
        rr_mean = rri.mean()
        rr_median = np.median(rri)
        rmssd = (rri_diff ** 2).mean() ** 0.5

        out = out.append([rr_min, rr_max, rr_minmax, rr_mean, rr_median, rmssd],
                         ['rr_min', 'rr_max', 'rr_minmax', 'rr_mean',
                          'rr_median', 'rmssd'])

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
        out_geom = compute_geometrical(rri=rri, show=show)

        out = out.join(out_geom)

    return out


def hrv_frequencydomain(rri=None, duration=None, freq_method='FFT',
                        fbands=None, detrend_rri=True, show=False, **kwargs):
    """Computes the frequency domain HRV features from a sequence of RR
    intervals.

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
    detrend_rri : bool, optional
        Whether to detrend the input signal. Default: True.
    show : bool, optional
        Whether to show the power spectrum plot. Default: False.
    kwargs : dict, optional
        frs : resampling frequency for the RRI sequence (Hz).
        nperseg : Length of each segment in Welch periodogram.
        nfft : Length of the FFT used in Welch function.

    Returns
    -------
    {fbands}_peak : float
        Peak frequency for each frequency band (Hz).
    {fbands}_pwr : float
        Absolute power for each frequency band (ms^2).
    {fbands}_rpwr : float
        Relative power for each frequency band (nu).
    lf_hf : float
        Ratio of LF-to-HF power.
    lf_nu : float
        Ratio of LF to LF+HF power (nu).
    hf_nu :  float
        Ratio of HF to LF+HF power (nu).
    total_pwr : float
        Total power.
    """

    # check inputs
    if rri is None:
        raise TypeError("Please specify an RRI list or array.")

    freq_methods = ['FFT']
    if freq_method not in freq_methods:
        raise ValueError(f"'{freq_method}' is not an available input. Choose"
                         f"one from: {freq_methods}.")

    if fbands is None:
        fbands = FBANDS

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # ensure minimal duration
    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 20:
        raise ValueError("Signal duration must be greater than 20 seconds to "
                         "compute frequency-domain features.")

    # initialize outputs
    out = utils.ReturnTuple((), ())
    out = out.append(fbands, 'fbands')

    # resampling with cubic interpolation for equidistant samples
    frs = kwargs['frs'] if 'frs' in kwargs else 4
    t = np.cumsum(rri)
    t -= t[0]
    rri_inter = interp1d(t, rri, 'cubic')
    t_inter = np.arange(t[0], t[-1], 1000. / frs)
    rri_inter = rri_inter(t_inter)

    # detrend
    if detrend_rri:
        rri_inter = detrend_window(rri_inter)['rri_det']
        print('Frequency domain: the rri sequence was detrended.')

    if duration >= 20:

        # compute frequencies and powers
        if freq_method == 'FFT':
            nperseg = kwargs['nperseg'] if 'nperseg' in kwargs else int(len(rri_inter)/4.5)
            nfft = kwargs['nfft'] if 'nfft' in kwargs else (256 if nperseg < 256 else 2**np.ceil(np.log(nperseg)/np.log(2)))

            frequencies, powers = welch(rri_inter, fs=frs, scaling='density',
                                        nperseg=nperseg, nfft=nfft)

            # add to output
            out = out.append([frequencies, powers, freq_method],
                             ['frequencies', 'powers', 'freq_method'])

        # compute frequency bands
        fb_out = compute_fbands(frequencies=frequencies, powers=powers, show=False)

        out = out.join(fb_out)

        # compute LF/HF ratio
        lf_hf = fb_out['lf_pwr'] / fb_out['hf_pwr']

        out = out.append(lf_hf, 'lf_hf')

        # compute LF and HF power in normal units
        lf_nu = fb_out['lf_pwr'] / (fb_out['lf_pwr'] + fb_out['hf_pwr'])
        hf_nu = 1 - lf_nu

        out = out.append([lf_nu, hf_nu], ['lf_nu', 'hf_nu'])

        # plot
        if show:
            legends = {'LF/HF': lf_hf}
            for key in out.keys():
                if key.endswith('_rpwr'):
                    legends[key] = out[key]

            plotting.plot_hrv_fbands(frequencies, powers, fbands, freq_method,
                                     legends, show=show)

    return out


def hrv_nonlinear(rri=None, duration=None, detrend_rri=True, show=False):
    """Computes the non-linear HRV features from a sequence of RR intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    detrend_rri : bool, optional
        Whether to detrend the input signal. Default: True.
    show : bool, optional
        Controls the plotting calls. Default: False.

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
    sd21 : float
        SD2/SD1 - SD2 to SD1 ratio.
    sampen : float
        Sample entropy.
    appen : float
        Approximate entropy.
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
        raise ValueError("Signal duration must be greater than 90 seconds to "
                         "compute non-linear features.")

    # detrend
    if detrend_rri:
        rri = detrend_window(rri)['rri_det']
        print('Non-linear domain: the rri sequence was detrended.')

    # initialize outputs
    out = utils.ReturnTuple((), ())

    if duration >= 90:
        # compute SD1, SD2, SD1/SD2 and S
        cp = compute_poincare(rri=rri, show=show)
        out = out.join(cp)

        # compute sample entropy
        sampen = sample_entropy(rri)
        out = out.append(sampen, 'sampen')

    if len(rri) >= 800 or duration == np.inf:
        # compute approximate entropy
        appen = approximate_entropy(rri)
        out = out.append(appen, 'appen')

    return out


def compute_fbands(frequencies, powers, fbands=None, method_name=None,
                   show=False):
    """Computes frequency domain features for the specified frequency bands.

    Parameters
    ----------
    frequencies : array
        Frequency axis.
    powers : array
        Power spectrum values for the frequency axis.
    fbands : dict, optional
        Dictionary containing the limits of the frequency bands.
    method_name : str, optional
        Method that was used to compute the power spectrum. Default: None.
    show : bool, optional
        Whether to show the power spectrum plot. Default: False.

    Returns
    -------
    {fbands}_peak : float
        Peak frequency of the frequency band (Hz).
    {fbands}_pwr : float
        Absolute power of the frequency band (ms^2).
    {fbands}_rpwr : float
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

        out = out.append([pwr, peak, rpwr], [fband + '_pwr', fband + '_peak',
                                             fband + '_rpwr'])

    # plot
    if show:
        # legends
        freq_legends = {}
        for key in out.keys():
            if key.endswith('_rpwr'):
                freq_legends[key] = out[key]

        plotting.plot_hrv_fbands(frequencies=frequencies,
                                 powers=powers,
                                 fbands=fbands,
                                 method_name=method_name,
                                 legends=freq_legends,
                                 show=show)

    return out


def compute_poincare(rri, show=False):
    """Compute the Poincaré features from a sequence of RR intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    show : bool, optional
        If True, show the Poincaré plot.

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
    sd21 : float
        SD2/SD1 - SD2 to SD1 ratio.
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

    # compute sd1/sd2 and sd2/sd1 ratio
    sd12 = sd1 / sd2
    sd21 = sd2 / sd1

    # output
    out = out.append([s, sd1, sd2, sd12, sd21], ['s', 'sd1', 'sd2', 'sd12',
                                                 'sd21'])

    if show:
        legends = {'SD1/SD2': sd12, 'SD2/SD1': sd21}
        plotting.plot_poincare(rri=rri,
                               s=s,
                               sd1=sd1,
                               sd2=sd2,
                               legends=legends,
                               show=show)

    return out


def compute_geometrical(rri, binsize=1/128, show=False):
    """Computes the geometrical features from a sequence of RR intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    binsize : float, optional
        Binsize for RRI histogram (s). Default: 1/128 s.
    show : bool, optional
        If True, show the RRI histogram. Default: False.

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
                               q_hist=q_hist,
                               hti=hti,
                               tinn=tinn,
                               show=show)

    # output
    out = utils.ReturnTuple([hti, tinn, bins, q_hist], ['hti', 'tinn',
                                                        'bins', 'q_hist'])

    return out


def detrend_window(rri, win_len=2000, **kwargs):
    """Facilitates RRI detrending method using a signal window.

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
    rri_trend : array
        Trend of the RRI signal.

    """

    # check input type
    win_len = int(win_len)

    # extract parameters
    smoothing_factor = kwargs['smoothing_factor'] if 'smoothing_factor' in kwargs else 500

    # detrend signal
    if len(rri) > win_len:
        # split the signal
        splits = int(len(rri)/win_len)
        rri_splits = np.array_split(rri, splits)

        # compute the detrended signal for each split
        rri_det = []
        for split in rri_splits:
            split_det = st.detrend_smoothness_priors(split, smoothing_factor)['detrended']
            rri_det.append(split_det)

        # concantenate detrended splits
        rri_det = np.concatenate(rri_det)
        rri_trend = None
    else:
        rri_det, rri_trend = st.detrend_smoothness_priors(rri, smoothing_factor)

    # output
    out = utils.ReturnTuple([rri_det, rri_trend], ['rri_det', 'rri_trend'])
    return out


def sample_entropy(rri, m=2, r=0.2):
    """Computes the sample entropy of an RRI sequence.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    m : int, optional
        Embedding dimension. Default: 2.
    r : int, float, optional
        Tolerance. It is then multiplied by the sequence standard deviation.
        Default: 0.2.

    Returns
    -------
    sampen :  float
        Sample entropy of the RRI sequence.

    References
    ----------
    https://en.wikipedia.org/wiki/Sample_entropy
    """

    # redefine r
    r = r * rri.std()

    n = len(rri)

    # Split time series and save all templates of length m
    xmi = np.array([rri[i: i + m] for i in range(n - m)])
    xmj = np.array([rri[i: i + m] for i in range(n - m + 1)])

    # Save all matches minus the self-match, compute B
    b = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([rri[i: i + m] for i in range(n - m + 1)])

    a = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(a / b)


def approximate_entropy(rri, m=2, r=0.2):
    """Computes the approximate entropy of an RRI sequence.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    m : int, optional
        Embedding dimension. Default: 2.
    r : int, float, optional
        Tolerance. It is then multiplied by the sequence standard deviation.
        Default: 0.2.

    Returns
    -------
    appen :  float
        Approximate entropy of the RRI sequence.

    References
    ----------
    https://en.wikipedia.org/wiki/Approximate_entropy
    """

    # redefine r
    r = r * rri.std()

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[rri[j] for j in range(i, i + m - 1 + 1)] for i in range(n - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (n - m + 1.0)
            for x_i in x
        ]
        return (n - m + 1.0) ** (-1) * sum(np.log(C))

    n = len(rri)

    return _phi(m) - _phi(m + 1)
