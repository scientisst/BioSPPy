# -*- coding: utf-8 -*-
"""
biosppy.plotting
----------------

This module provides utilities to plot data.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range, zip
import six

# built-in
import os

# 3rd party
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# local
from . import utils
from biosppy.signals import tools as st

# Globals
MAJOR_LW = 1.5
MED_LW = 1.25
MINOR_LW = 1.0
MAX_ROWS = 10
LOGOS_FOLDER = '.logos'
BIOSPPY_LOGO = 'biosppy.png'
SCIENTISST_LOGO = 'scientisst.png'

# matplotlib settings
plt.rcParams['text.color'] = '#495057'
plt.rcParams['axes.labelcolor'] = '#495057'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.edgecolor'] = '#495057'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.color'] = '#495057'
plt.rcParams['ytick.color'] = '#495057'
plt.rcParams['grid.color'] = '#ebeef0'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.color'] = '#ebeef0'
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.framealpha'] = 0.75
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['legend.fontsize'] = 8


def color_palette(idx):
    """Color palette to use throughout the biosppy package

    Parameters
    ----------
    idx: str or int
        identifier of color to use

    Returns
    -------
    color_id: str
        hexadecimal color code chosen
    """

    color_dict = {
        'blue': '#71a7cc',
        'dark-blue': '#184169',
        'light-blue': '#a5c8e0',

        'green': '#329352',
        'dark-green': '#154d28',
        'light-green': '#7ac77d',

        'red': '#D62839',
        'dark-red': '#B14343',
        'light-red': '#FF5A5F',

        'yellow': '#D19C2F',
        'dark-yellow': '#FFC300',
        'light-yellow': '#FFDD55',

        'violet': '#9D4EDD',
        'dark-violet': '#3C096C',
        'light-violet': '#E0AAFF',

        'orange': '#f3883e',
        'dark-orange': '#b2400a',
        'light-orange': '#f7d1ab',

        'grey': '#ADB5BD',
        'dark-grey': '#495057',
        'light-grey': '#E6E6E6'
    }

    if type(idx) == int:
        color_id = list(color_dict.values())[idx]
    else:
        if idx in color_dict.keys():
            color_id = color_dict[idx]
        else:
            raise ValueError(f'Please choose one color from {color_dict.keys()}'
                             f' or give an index')

    return color_id


def add_logo(fig):
    logos_folder = os.path.join(os.path.dirname(__file__), LOGOS_FOLDER)
    try:
        # add biosppy logo
        biosppy_logo = plt.imread(os.path.join(logos_folder, BIOSPPY_LOGO))
        logo = fig.add_axes([0.80, 0.02, 0.08, 0.08], anchor='SE')
        logo.imshow(biosppy_logo, alpha=0.5)
        logo.axis('off')

        # add scientisst logo
        scientisst_logo = plt.imread(os.path.join(logos_folder, SCIENTISST_LOGO))
        logo = fig.add_axes([0.90, 0.02, 0.08, 0.08], anchor='SE')
        logo.imshow(scientisst_logo, alpha=0.5)
        logo.axis('off')
    except:
        pass


def _plot_filter(b, a, sampling_rate=1000., nfreqs=4096, log_xscale=True,
                 ax=None):
    """Compute and plot the frequency response of a digital filter.

    Parameters
    ----------
    b : array
        Numerator coefficients.
    a : array
        Denominator coefficients.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    nfreqs : int, optional
        Number of frequency points to compute.
    log_xscale : bool, optional
        Whether to use log scale for x-axis.
    ax : axis, optional
        Plot Axis to use.

    Returns
    -------
    fig : Figure
        Figure object.

    """

    # compute frequency response
    freqs, resp = st._filter_resp(b, a,
                                  sampling_rate=sampling_rate,
                                  nfreqs=nfreqs)

    # plot
    if ax is None:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    else:
        ax2 = ax.twinx()

    # set layout
    fig.subplots_adjust(top=0.88, bottom=0.17, hspace=0.2, left=0.16,
                        right=0.94)

    # title
    fig.suptitle('Filter Frequency Response', fontsize=12, fontweight='bold')

    # amplitude
    pwr = 20. * np.log10(np.abs(resp))
    if log_xscale:
        ax.semilogx(freqs, pwr, color=color_palette('blue'),
                    linewidth=MAJOR_LW)
    else:
        ax.plot(freqs, pwr, color=color_palette('blue'), linewidth=MAJOR_LW)
    ax.set_ylabel('Amplitude (dB)')
    ax.grid(True)
    ax.tick_params(axis='x', which='both', bottom=False, top=False,
                   labelbottom=False)
    ax.spines['bottom'].set_visible(False)

    # phase
    angles = np.unwrap(np.angle(resp)) * 180. / np.pi  # radians to degrees
    if log_xscale:
        ax2.semilogx(freqs, angles, color=color_palette('dark-blue'),
                     linewidth=MAJOR_LW)
    else:
        ax2.plot(freqs, angles, color=color_palette('dark-blue'),
                 linewidth=MAJOR_LW)
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.grid(True)

    # align y-axis labels
    fig.align_ylabels()

    # add logo
    add_logo(fig)

    return fig


def plot_filter(ftype='FIR',
                band='lowpass',
                order=None,
                frequency=None,
                sampling_rate=1000.,
                log_xscale=True,
                path=None,
                show=True, **kwargs):
    """Plot the frequency response of the filter specified with the given
    parameters.

    Parameters
    ----------
    ftype : str
        Filter type:
            * Finite Impulse Response filter ('FIR');
            * Butterworth filter ('butter');
            * Chebyshev filters ('cheby1', 'cheby2');
            * Elliptic filter ('ellip');
            * Bessel filter ('bessel').
    band : str
        Band type:
            * Low-pass filter ('lowpass');
            * High-pass filter ('highpass');
            * Band-pass filter ('bandpass');
            * Band-stop filter ('bandstop').
    order : int
        Order of the filter.
    frequency : int, float, list, array
        Cutoff frequencies; format depends on type of band:
            * 'lowpass' or 'bandpass': single frequency;
            * 'bandpass' or 'bandstop': pair of frequencies.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    log_xscale : bool, optional
        Whether to use log scale for x-axis.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.
    ``**kwargs`` : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal function.

    """

    # get filter
    b, a = st.get_filter(ftype=ftype,
                         band=band,
                         order=order,
                         frequency=frequency,
                         sampling_rate=sampling_rate, **kwargs)

    # plot
    fig = _plot_filter(b, a, sampling_rate, log_xscale=log_xscale)

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def plot_spectrum(signal=None, sampling_rate=1000., path=None, show=True):
    """Plot the power spectrum of a signal (one-sided).

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    freqs, power = st.power_spectrum(signal, sampling_rate,
                                     pad=0,
                                     pow2=False,
                                     decibel=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(freqs, power, linewidth=MAJOR_LW)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)')
    ax.grid()

    # make layout tight
    fig.tight_layout()

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def plot_acc(ts=None,
             raw=None,
             vm=None,
             sm=None,
             units=None,
             path=None,
             show=False):
    """Create a summary plot from the output of signals.acc.acc.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw ACC signal.
    vm : array
        Vector Magnitude feature of the signal.
    sm : array
        Signal Magnitude feature of the signal
    units : str, optional
        Units of the vertical axis. If provided, the plot title will include
        the units information. Default is None.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    raw_t = np.transpose(raw)
    acc_x, acc_y, acc_z = raw_t[0], raw_t[1], raw_t[2]

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('ACC Summary', fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(6, 2)
    fig.subplots_adjust(top=0.85, hspace=0.7, wspace=0.34, left=0.13,
                        right=0.96, bottom=0.18)

    # raw signal (acc_x)
    ax1 = fig.add_subplot(gs[:2, 0])
    ax1.set_title("Signal")

    ax1.plot(ts, acc_x, linewidth=MINOR_LW, label='X',
             color=color_palette('dark-blue'))

    ax1.legend(loc='upper right')
    ax1.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax1.spines['bottom'].set_visible(False)

    # raw signal (acc_y)
    ax2 = fig.add_subplot(gs[2:4, 0], sharex=ax1)

    ax2.plot(ts, acc_y, linewidth=MINOR_LW, label='Y',
             color=color_palette('blue'))

    ax2.set_ylabel('Amplitude' if units is None else 'Amplitude ($%s$)' % units)
    ax2.legend(loc='upper right')
    ax2.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax2.spines['bottom'].set_visible(False)

    # raw signal (acc_z)
    ax3 = fig.add_subplot(gs[4:, 0], sharex=ax1)

    ax3.plot(ts, acc_z, linewidth=MINOR_LW, label='Z',
             color=color_palette('light-blue'))

    ax3.set_xlabel('Time (s)')
    ax3.legend(loc='upper right')

    # vector magnitude
    ax4 = fig.add_subplot(gs[:3, 1], sharex=ax1)
    ax4.set_title('Features')

    ax4.plot(ts, vm, linewidth=MINOR_LW, label='Vector Magnitude',
             color=color_palette('orange'))

    ax4.set_ylabel('Amplitude' if units is None else 'Amplitude ($%s$)' % units)
    ax4.legend(loc='upper right')
    ax4.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax4.spines['bottom'].set_visible(False)

    # signal magnitude
    ax5 = fig.add_subplot(gs[3:, 1], sharex=ax1)

    ax5.plot(ts, sm, linewidth=MINOR_LW, label='Signal Magnitude',
             color=color_palette('light-orange'))

    ax5.set_ylabel('Amplitude' if units is None else 'Amplitude (%s)' % units)
    ax5.set_xlabel('Time (s)')
    ax5.legend(loc='upper right')

    # align y-axis labels
    fig.align_ylabels([ax1, ax2, ax3])
    fig.align_ylabels([ax4, ax5])

    # add logo
    add_logo(fig)

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def plot_ppg(ts=None,
             raw=None,
             filtered=None,
             peaks=None,
             templates_ts=None,
             templates=None,
             heart_rate_ts=None,
             heart_rate=None,
             units=None,
             path=None,
             show=False):
    """Create a summary plot from the output of signals.ppg.ppg.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw PPG signal.
    filtered : array
        Filtered PPG signal.
    peaks : array
        Indices of PPG pulse peaks.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted PPG templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).
    units : str, optional
        Units of the vertical axis. If provided, the plot title will include
        the units information. Default is None.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('PPG Summary', fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(6, 2)
    fig.subplots_adjust(top=0.88, bottom=0.12, hspace=0.9, wspace=0.3,
                        left=0.1, right=0.96)

    # raw signal
    ax1 = fig.add_subplot(gs[:2, 0])
    ax1.set_title('Signal')

    ax1.plot(ts, raw, linewidth=MED_LW, label='Raw',
             color=color_palette('light-blue'), alpha=0.6)
    ax1.plot(ts, filtered+np.mean(raw), linewidth=MINOR_LW, label='Filtered',
             color=color_palette('blue'))

    ax1.set_ylabel('Amplitude' if units is None else 'Amplitude (%s)' % units)
    ax1.legend()
    ax1.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax1.spines['bottom'].set_visible(False)

    #  signal with onsets
    ax2 = fig.add_subplot(gs[2:4, 0], sharex=ax1)
    ax2.set_title('Systolic Peak Detection')

    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    ax2.plot(ts, filtered, linewidth=MINOR_LW, color=color_palette('blue'))
    ax2.plot(ts[peaks], filtered[peaks], ls='None', marker='x',
             color=color_palette('dark-red'), label='Peaks')

    ax2.set_ylabel('Amplitude' if units is None else 'Amplitude (%s)' % units)
    ax2.legend()

    ax2.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax2.spines['bottom'].set_visible(False)

    # heart rate
    ax3 = fig.add_subplot(gs[4:, 0], sharex=ax1)
    ax3.set_title('Heart Rate')

    ax3.plot(heart_rate_ts, heart_rate, linewidth=MAJOR_LW, label='Heart Rate',
             color=color_palette('blue'))

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Heart Rate (bpm)')

    # align y-axis labels
    fig.align_ylabels([ax1, ax2, ax3])

    # templates
    ax4 = fig.add_subplot(gs[1:5, 1])

    ax4.plot(templates_ts, templates, linewidth=MINOR_LW, alpha=0.5,
             color=color_palette('blue'))

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude' if units is None else 'Amplitude (%s)' % units)
    ax4.set_title('Templates')

    add_logo(fig)

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def plot_bvp(ts=None,
             raw=None,
             filtered=None,
             onsets=None,
             heart_rate_ts=None,
             heart_rate=None,
             path=None,
             show=False):
    """Create a summary plot from the output of signals.bvp.bvp.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw BVP signal.
    filtered : array
        Filtered BVP signal.
    onsets : array
        Indices of BVP pulse onsets.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure()
    fig.suptitle('BVP Summary')

    # raw signal
    ax1 = fig.add_subplot(311)

    ax1.plot(ts, raw, linewidth=MAJOR_LW, label='Raw')

    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid()

    # filtered signal with onsets
    ax2 = fig.add_subplot(312, sharex=ax1)

    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    ax2.plot(ts, filtered, linewidth=MAJOR_LW, label='Filtered')
    ax2.vlines(ts[onsets], ymin, ymax,
               color='m',
               linewidth=MINOR_LW,
               label='Onsets')

    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid()

    # heart rate
    ax3 = fig.add_subplot(313, sharex=ax1)

    ax3.plot(heart_rate_ts, heart_rate, linewidth=MAJOR_LW, label='Heart Rate')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Heart Rate (bpm)')
    ax3.legend()
    ax3.grid()

    # make layout tight
    fig.tight_layout()

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def plot_abp(ts=None,
             raw=None,
             filtered=None,
             onsets=None,
             heart_rate_ts=None,
             heart_rate=None,
             path=None,
             show=False):
    """Create a summary plot from the output of signals.abp.abp.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw ABP signal.
    filtered : array
        Filtered ABP signal.
    onsets : array
        Indices of ABP pulse onsets.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure()
    fig.suptitle('ABP Summary')

    # raw signal
    ax1 = fig.add_subplot(311)

    ax1.plot(ts, raw, linewidth=MAJOR_LW, label='Raw')

    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid()

    # filtered signal with onsets
    ax2 = fig.add_subplot(312, sharex=ax1)

    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    ax2.plot(ts, filtered, linewidth=MAJOR_LW, label='Filtered')
    ax2.vlines(ts[onsets], ymin, ymax,
               color='m',
               linewidth=MINOR_LW,
               label='Onsets')

    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid()

    # heart rate
    ax3 = fig.add_subplot(313, sharex=ax1)

    ax3.plot(heart_rate_ts, heart_rate, linewidth=MAJOR_LW, label='Heart Rate')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Heart Rate (bpm)')
    ax3.legend()
    ax3.grid()

    # make layout tight
    fig.tight_layout()

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def plot_eda(ts=None,
             raw=None,
             filtered=None,
             edr=None,
             edl=None,
             onsets=None,
             peaks=None,
             amplitudes=None,
             units=None,
             path=None,
             show=False):
    """Create a summary plot from the output of signals.eda.eda.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw EDA signal.
    filtered : array
        Filtered EDA signal.
    edr : array
        Electrodermal response signal.
    edl : array
        Electrodermal level signal.
    onsets : array
        Events onsets indices.
    peaks : array
        Events peaks indices.
    amplitudes : array
        Amplitudes location indices.
    units : str, optional
        Units of the vertical axis. If provided, the plot title will include
        the units information. Default is None.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('EDA Summary', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(6, 2)
    fig.subplots_adjust(top=0.85, hspace=0.7, wspace=0.34, left=0.13,
                        right=0.96, bottom=0.18)

    # signal
    ax1 = fig.add_subplot(gs[:2, 0])

    ax1.plot(ts, raw, linewidth=MED_LW, label='Raw',
             color=color_palette('light-blue'), alpha=0.6)
    ax1.plot(ts, filtered, linewidth=MINOR_LW, label='Filtered',
             color=color_palette('blue'))

    ax1.set_ylabel('Amplitude' if units is None else 'Amplitude \n (%s)' % units)
    ax1.set_title('Signal')
    ax1.legend()

    ax1.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax1.spines['bottom'].set_visible(False)

    # event detection
    ax2 = fig.add_subplot(gs[2:4, 0], sharex=ax1, sharey=ax1)

    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    ax2.plot(ts, filtered, linewidth=MINOR_LW, color=color_palette('blue'))
    ax2.scatter(ts[onsets], filtered[onsets], marker='|', lw=1, s=100,
                color=color_palette('dark-red'), label='Onsets', zorder=3)
    ax2.scatter(ts[peaks], filtered[peaks], marker='x',
                color=color_palette('dark-red'), label='Peaks', zorder=3)

    ax2.set_ylabel('Amplitude' if units is None else 'Amplitude \n (%s)' % units)
    ax2.set_title('Event Detection')
    ax2.legend()

    ax2.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax2.spines['bottom'].set_visible(False)

    # amplitudes
    ax3 = fig.add_subplot(gs[4:, 0], sharex=ax1)

    ax3.plot(ts[peaks], amplitudes, linewidth=MAJOR_LW, color=color_palette('blue'), label='Amplitude')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude' if units is None else 'Amplitude \n (%s)' % units)
    ax3.set_title('Event Amplitudes')

    # align y axis labels
    fig.align_ylabels()

    # decomposition EDL
    ax4 = fig.add_subplot(gs[0:3, 1], sharex=ax1, sharey=ax1)

    ax4.plot(ts, filtered, linewidth=MAJOR_LW, color=color_palette('light-blue'))
    ax4.plot(ts, edl, linewidth=MINOR_LW, color=color_palette('dark-orange'),
             label="EDL")

    ax4.set_ylabel('Amplitude' if units is None else 'Amplitude \n (%s)' % units)
    ax4.set_title('EDA Decomposition')
    ax4.legend()

    ax4.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax4.spines['bottom'].set_visible(False)

    # decomposition EDR
    ax5 = fig.add_subplot(gs[3:, 1], sharex=ax1)

    ax5.plot(ts[1:], edr, linewidth=MINOR_LW,
             color=color_palette('dark-orange'), label="EDR")

    ax5.set_ylabel('Amplitude' if units is None else 'Amplitude \n (%s)' % units)
    ax5.set_xlabel('Time (s)')
    ax5.legend()

    # logo
    add_logo(fig)

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def plot_emg(ts=None,
             sampling_rate=None,
             raw=None,
             filtered=None,
             onsets=None,
             processed=None,
             units=None,
             path=None,
             show=False):
    """Create a summary plot from the output of signals.emg.emg.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    sampling_rate : int, float
        Sampling frequency (Hz).
    raw : array
        Raw EMG signal.
    filtered : array
        Filtered EMG signal.
    onsets : array
        Indices of EMG pulse onsets.
    processed : array, optional
        Processed EMG signal according to the chosen onset detector.
    units : str, optional
        Units of the vertical axis. If provided, the plot title will include
        the units information. Default is None.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('EMG Summary', fontsize=12, fontweight='bold')
    fig.subplots_adjust(top=0.85, hspace=0.25, wspace=0.34, left=0.1,
                        right=0.96, bottom=0.18)

    if processed is not None:
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312, sharex=ax1)
        ax3 = fig.add_subplot(313)

        # processed signal
        L = len(processed)
        T = (L - 1) / sampling_rate
        ts_processed = np.linspace(0, T, L, endpoint=True)
        ax3.plot(ts_processed, processed,
                 linewidth=MINOR_LW,
                 label='Processed',
                 color=color_palette('blue'))
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.legend(loc='upper right')
    else:
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

    # signal
    ax1.set_title('Signal')
    ax1.plot(ts, raw, linewidth=MED_LW, label='Raw', alpha=0.6,
             color=color_palette('light-blue'))
    ax1.plot(ts, filtered+np.mean(raw), linewidth=MINOR_LW, label='Filtered',
             color=color_palette('blue'))

    ax1.set_ylabel('Amplitude' if units is None else 'Amplitude (%s)' % units)
    ax1.legend(loc='upper right')
    ax1.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax1.spines['bottom'].set_visible(False)

    # filtered signal with onsets
    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    ax2.set_title('Event Detection')
    ax2.plot(ts, filtered, linewidth=MINOR_LW, color=color_palette('blue'))
    ax2.vlines(ts[onsets], ymin, ymax,
               color=color_palette('orange'),
               linewidth=MINOR_LW,
               label='Onsets')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude' if units is None else 'Amplitude (%s)' % units)
    ax2.legend(loc='upper right')

    # align y axis labels
    fig.align_ylabels()

    # add logo
    add_logo(fig)

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def plot_resp(ts=None,
              raw=None,
              filtered=None,
              zeros=None,
              resp_rate_ts=None,
              resp_rate=None,
              units=None,
              path=None,
              show=False):
    """Create a summary plot from the output of signals.ppg.ppg.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw Resp signal.
    filtered : array
        Filtered Resp signal.
    zeros : array
        Indices of Respiration zero crossings.
    resp_rate_ts : array
        Respiration rate time axis reference (seconds).
    resp_rate : array
        Instantaneous respiration rate (Hz).
    units : str, optional
        Units of the vertical axis. If provided, the plot title will include
        the units information. Default is None.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('RESP Summary', fontsize=12, fontweight='bold')
    fig.subplots_adjust(top=0.85, hspace=0.35, wspace=0.34, left=0.13,
                        right=0.96, bottom=0.18)

    # raw signal
    ax1 = fig.add_subplot(311)
    ax1.set_title('Signal')

    ax1.plot(ts, raw, linewidth=MED_LW, label='Raw', alpha=0.6,
             color=color_palette('light-blue'))
    ax1.plot(ts, filtered+np.mean(raw), linewidth=MINOR_LW, label='Filtered',
                color=color_palette('blue'))

    ax1.set_ylabel('Amplitude' if units is None else 'Amplitude \n (%s)' % units)
    ax1.legend(loc='upper right')
    ax1.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax1.spines['bottom'].set_visible(False)

    # filtered signal with zeros
    ax2 = fig.add_subplot(312, sharex=ax1)

    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    ax2.plot(ts, filtered, linewidth=MINOR_LW, color=color_palette('blue'))
    ax2.vlines(ts[zeros], ymin, ymax,
               color=color_palette('dark-red'),
               linewidth=MINOR_LW,
               label='Zero crossings')

    ax2.set_ylabel('Amplitude' if units is None else 'Amplitude \n (%s)' % units)
    ax2.legend(loc='upper right')
    ax2.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax2.spines['bottom'].set_visible(False)

    # respiration rate
    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.set_title('Respiration Rate')

    ax3.plot(resp_rate_ts, resp_rate, linewidth=MAJOR_LW,
             color=color_palette('blue'))

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Respiration Rate \n(Hz)')

    # align y axes labels
    fig.align_ylabels()

    # add logo
    add_logo(fig)

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def plot_eeg(ts=None,
             raw=None,
             filtered=None,
             labels=None,
             features_ts=None,
             theta=None,
             alpha_low=None,
             alpha_high=None,
             beta=None,
             gamma=None,
             plf_pairs=None,
             plf=None,
             path=None,
             show=False):
    """Create a summary plot from the output of signals.eeg.eeg.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw EEG signal.
    filtered : array
        Filtered EEG signal.
    labels : list
        Channel labels.
    features_ts : array
        Features time axis reference (seconds).
    theta : array
        Average power in the 4 to 8 Hz frequency band; each column is one
        EEG channel.
    alpha_low : array
        Average power in the 8 to 10 Hz frequency band; each column is one
        EEG channel.
    alpha_high : array
        Average power in the 10 to 13 Hz frequency band; each column is one
        EEG channel.
    beta : array
        Average power in the 13 to 25 Hz frequency band; each column is one
        EEG channel.
    gamma : array
        Average power in the 25 to 40 Hz frequency band; each column is one
        EEG channel.
    plf_pairs : list
        PLF pair indices.
    plf : array
        PLF matrix; each column is a channel pair.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    nrows = MAX_ROWS
    alpha = 2.

    # Get number of channels
    nch = raw.shape[1]

    figs = []

    # raw
    fig = _plot_multichannel(ts=ts,
                             signal=raw,
                             labels=labels,
                             nrows=nrows,
                             alpha=alpha,
                             title='EEG Summary - Raw',
                             xlabel='Time (s)',
                             ylabel='Amplitude')
    figs.append(('_Raw', fig))

    # filtered
    fig = _plot_multichannel(ts=ts,
                             signal=filtered,
                             labels=labels,
                             nrows=nrows,
                             alpha=alpha,
                             title='EEG Summary - Filtered',
                             xlabel='Time (s)',
                             ylabel='Amplitude')
    figs.append(('_Filtered', fig))

    # band-power
    names = ('Theta Band', 'Lower Alpha Band', 'Higher Alpha Band',
             'Beta Band', 'Gamma Band')
    args = (theta, alpha_low, alpha_high, beta, gamma)
    for n, a in zip(names, args):
        fig = _plot_multichannel(ts=features_ts,
                                 signal=a,
                                 labels=labels,
                                 nrows=nrows,
                                 alpha=alpha,
                                 title='EEG Summary - %s' % n,
                                 xlabel='Time (s)',
                                 ylabel='Power')
        figs.append(('_' + n.replace(' ', '_'), fig))

    # Only plot/compute plf if there is more than one channel
    if nch > 1:
        # PLF
        plf_labels = ['%s vs %s' % (labels[p[0]], labels[p[1]]) for p in plf_pairs]
        fig = _plot_multichannel(ts=features_ts,
                                 signal=plf,
                                 labels=plf_labels,
                                 nrows=nrows,
                                 alpha=alpha,
                                 title='EEG Summary - Phase-Locking Factor',
                                 xlabel='Time (s)',
                                 ylabel='PLF')
        figs.append(('_PLF', fig))

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            ext = '.png'

        for n, fig in figs:
            path = root + n + ext
            fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        for _, fig in figs:
            plt.close(fig)


def _yscaling(signal=None, alpha=1.5):
    """Get y axis limits for a signal with scaling.

    Parameters
    ----------
    signal : array
        Input signal.
    alpha : float, optional
        Scaling factor.

    Returns
    -------
    ymin : float
        Minimum y value.
    ymax : float
        Maximum y value.

    """

    mi = np.min(signal)
    m = np.mean(signal)
    mx = np.max(signal)

    if mi == mx:
        ymin = m - 1
        ymax = m + 1
    else:
        ymin = m - alpha * (m - mi)
        ymax = m + alpha * (mx - m)

    return ymin, ymax


def _plot_multichannel(ts=None,
                       signal=None,
                       labels=None,
                       nrows=10,
                       alpha=2.,
                       title=None,
                       xlabel=None,
                       ylabel=None):
    """Plot a multi-channel signal.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    signal : array
        Multi-channel signal; each column is one channel.
    labels : list, optional
        Channel labels.
    nrows : int, optional
        Maximum number of rows to use.
    alpha : float, optional
        Scaling factor for y axis.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for x axis.
    ylabel : str, optional
        Label for y axis.

    Returns
    -------
    fig : Figure
        Figure object.

    """

    # ensure numpy
    signal = np.array(signal)
    nch = signal.shape[1]

    # check labels
    if labels is None:
        labels = ['Ch. %d' % i for i in range(nch)]

    if nch < nrows:
        nrows = nch

    ncols = int(np.ceil(nch / float(nrows)))

    fig = plt.figure()

    # title
    if title is not None:
        fig.suptitle(title)

    gs = gridspec.GridSpec(nrows, ncols, hspace=0, wspace=0.2)

    # reference axes
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(ts, signal[:, 0], linewidth=MAJOR_LW, label=labels[0])
    ymin, ymax = _yscaling(signal[:, 0], alpha=alpha)
    ax0.set_ylim(ymin, ymax)
    ax0.legend()
    ax0.grid()
    axs = {(0, 0): ax0}

    for i in range(1, nch - 1):
        a = i % nrows
        b = int(np.floor(i / float(nrows)))
        ax = fig.add_subplot(gs[a, b], sharex=ax0)
        axs[(a, b)] = ax

        ax.plot(ts, signal[:, i], linewidth=MAJOR_LW, label=labels[i])
        ymin, ymax = _yscaling(signal[:, i], alpha=alpha)
        ax.set_ylim(ymin, ymax)
        ax.legend()
        ax.grid()

    # last plot
    i = nch - 1
    a = i % nrows
    b = int(np.floor(i / float(nrows)))
    ax = fig.add_subplot(gs[a, b], sharex=ax0)
    axs[(a, b)] = ax

    ax.plot(ts, signal[:, -1], linewidth=MAJOR_LW, label=labels[-1])
    ymin, ymax = _yscaling(signal[:, -1], alpha=alpha)
    ax.set_ylim(ymin, ymax)
    ax.legend()
    ax.grid()

    if xlabel is not None:
        ax.set_xlabel(xlabel)

        for b in range(0, ncols - 1):
            a = nrows - 1
            ax = axs[(a, b)]
            ax.set_xlabel(xlabel)

    if ylabel is not None:
        # middle left
        a = nrows // 2
        ax = axs[(a, 0)]
        ax.set_ylabel(ylabel)

    # make layout tight
    gs.tight_layout(fig)

    return fig


def plot_ecg(ts=None,
             raw=None,
             filtered=None,
             rpeaks=None,
             templates_ts=None,
             templates=None,
             heart_rate_ts=None,
             heart_rate=None,
             units=None,
             path=None,
             show=False):
    """Create a summary plot from the output of signals.ecg.ecg.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw ECG signal.
    filtered : array
        Filtered ECG signal.
    rpeaks : array
        R-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).
    units : str, optional
        Units of the vertical axis. If provided, the plot title will include
        the units information. Default is None.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('ECG Summary', fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(6, 2)
    fig.subplots_adjust(top=0.88, bottom=0.12, hspace=0.9, wspace=0.3,
                        left=0.1, right=0.96)

    # signal
    ax1 = fig.add_subplot(gs[:2, 0])
    ax1.set_title('Signal')

    ax1.plot(ts, raw, linewidth=MED_LW, label='Raw', alpha=0.6,
             color=color_palette('light-blue'))
    ax1.plot(ts, filtered+np.mean(raw), linewidth=MINOR_LW, label='Filtered',
                color=color_palette('blue'))

    ax1.set_ylabel('Amplitude' if units is None else 'Amplitude (%s)' % units)
    ax1.legend(loc='upper right')
    ax1.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax1.spines['bottom'].set_visible(False)

    # filtered signal with rpeaks
    ax2 = fig.add_subplot(gs[2:4, 0], sharex=ax1)
    ax2.set_title('R-Peak Detection')

    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    ax2.plot(ts, filtered, linewidth=MINOR_LW, color=color_palette('blue'))
    ax2.plot(ts[rpeaks], filtered[rpeaks], ls='None', marker='x',
             color=color_palette('dark-red'), label='Peaks')

    ax2.set_ylabel('Amplitude' if units is None else 'Amplitude (%s)' % units)
    ax2.legend(loc='upper right')
    ax2.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax2.spines['bottom'].set_visible(False)

    # heart rate
    ax3 = fig.add_subplot(gs[4:, 0], sharex=ax1)
    ax3.set_title('Heart Rate')

    ax3.plot(heart_rate_ts, heart_rate, linewidth=MAJOR_LW,
             color=color_palette('blue'))

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Heart Rate (bpm)')

    # align y axis labels
    fig.align_ylabels([ax1, ax2, ax3])

    # templates
    ax4 = fig.add_subplot(gs[1:5, 1])
    ax4.set_title('Templates')

    ax4.plot(templates_ts, templates.T, linewidth=MINOR_LW, alpha=0.5,
             color=color_palette('blue'))

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude' if units is None else 'Amplitude (%s)' % units)

    # add logo
    add_logo(fig)

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def plot_bcg(ts=None,
             raw=None,
             filtered=None,
             jpeaks=None,
             templates_ts=None,
             templates=None,
             heart_rate_ts=None,
             heart_rate=None,
             path=None,
             show=False):
    """Create a summary plot from the output of signals.bcg.bcg.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw ECG signal.
    filtered : array
        Filtered ECG signal.
    ipeaks : array
        I-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure()
    fig.suptitle('BCG Summary')
    gs = gridspec.GridSpec(6, 2)

    # raw signal
    ax1 = fig.add_subplot(gs[:2, 0])

    ax1.plot(ts, raw, linewidth=MAJOR_LW, label='Raw')

    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid()

    # filtered signal with rpeaks
    ax2 = fig.add_subplot(gs[2:4, 0], sharex=ax1)

    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    ax2.plot(ts, filtered, linewidth=MAJOR_LW, label='Filtered')
    ax2.vlines(ts[jpeaks], ymin, ymax,
               color='m',
               linewidth=MINOR_LW,
               label='J-peaks')

    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid()

    # heart rate
    ax3 = fig.add_subplot(gs[4:, 0], sharex=ax1)

    ax3.plot(heart_rate_ts, heart_rate, linewidth=MAJOR_LW, label='Heart Rate')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Heart Rate (bpm)')
    ax3.legend()
    ax3.grid()

    # templates
    ax4 = fig.add_subplot(gs[1:5, 1])

    ax4.plot(templates_ts, templates.T, 'm', linewidth=MINOR_LW, alpha=0.7)

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Templates')
    ax4.grid()

    # make layout tight
    gs.tight_layout(fig)

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def plot_pcg(ts=None,
             raw=None,
             filtered=None,
             peaks=None,
             heart_sounds=None,
             heart_rate_ts=None,
             inst_heart_rate=None,
             units=None,
             path=None,
             show=False):
    """Create a summary plot from the output of signals.pcg.pcg.
    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw PCG signal.
    filtered : array
        Filtered PCG signal.
    peaks : array
        Peak location indices.
    heart_sounds : array
        Classification of peaks as S1 or S2
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    inst_heart_rate : array
        Instantaneous heart rate (bpm).
    units : str, optional
        Units of the vertical axis. If provided, the plot title will include
        the units information. Default is None.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('PCG Summary', fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(6, 2)
    fig.subplots_adjust(top=0.88, bottom=0.12, hspace=0.9, wspace=0.3,
                        left=0.1, right=0.96)

    # signal
    ax1 = fig.add_subplot(gs[:2, 0])
    ax1.set_title('Signal')

    ax1.plot(ts, raw, linewidth=MED_LW, label='Raw',
             color=color_palette('light-blue'))
    ax1.plot(ts, filtered+np.mean(raw), linewidth=MINOR_LW, label='Filtered',
                color=color_palette('blue'))

    ax1.set_ylabel('Amplitude' if units is None else 'Amplitude (%s)' % units)
    ax1.legend(loc='upper right')
    ax1.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax1.spines['bottom'].set_visible(False)

    # filtered signal with sounds
    ax2 = fig.add_subplot(gs[2:4, 0], sharex=ax1)
    ax2.set_title('Heart Sound Detection')

    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    ax2.plot(ts, filtered, linewidth=MINOR_LW, color=color_palette('blue'))
    ax2.vlines(ts[peaks], ymin, ymax,
               color=color_palette('dark-red'),
               linewidth=MED_LW,
               label='Heart Sounds')

    ax2.set_ylabel('Amplitude' if units is None else 'Amplitude (%s)' % units)
    ax2.legend(loc='upper right')
    ax2.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax2.spines['bottom'].set_visible(False)

    # heart rate
    ax3 = fig.add_subplot(gs[4:, 0], sharex=ax1)
    ax3.set_title('Heart Rate')

    ax3.plot(heart_rate_ts, inst_heart_rate, linewidth=MAJOR_LW,
             color=color_palette('blue'))

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Heart Rate (bpm)')

    # align y axis labels
    fig.align_ylabels([ax1, ax2, ax3])

    # heart sounds
    ax4 = fig.add_subplot(gs[1:5, 1], sharex=ax1)

    ax4.plot(ts, filtered, linewidth=MINOR_LW, color=color_palette('blue'))
    for i in range(0, len(peaks)):
        text = "S" + str(int(heart_sounds[i]))
        plt.annotate(text, (ts[peaks[i]], ymax - alpha), ha='center', va='center', size=9,
                     color=color_palette('dark-grey'))

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude' if units is None else 'Amplitude (%s)' % units)
    ax4.set_title('Heart Sounds Classification')

    # add logo
    add_logo(fig)

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def _plot_rates(thresholds, rates, variables,
                lw=1,
                colors=None,
                alpha=1,
                eer_idx=None,
                labels=False,
                ax=None):
    """Plot biometric rates.

    Parameters
    ----------
    thresholds : array
        Classifier thresholds.
    rates : dict
        Dictionary of rates.
    variables : list
        Keys from 'rates' to plot.
    lw : int, float, optional
        Plot linewidth.
    colors : list, optional
        Plot line color for each variable.
    alpha : float, optional
        Plot line alpha value.
    eer_idx : int, optional
        Classifier reference index for the Equal Error Rate.
    labels : bool, optional
        If True, will show plot labels.
    ax : axis, optional
        Plot Axis to use.

    Returns
    -------
    fig : Figure
        Figure object.

    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if colors is None:
        x = np.linspace(0., 1., len(variables))
        colors = plt.get_cmap('rainbow')(x)

    if labels:
        for i, v in enumerate(variables):
            ax.plot(thresholds, rates[v], colors[i],
                    lw=lw,
                    alpha=alpha,
                    label=v)
    else:
        for i, v in enumerate(variables):
            ax.plot(thresholds, rates[v], colors[i], lw=lw, alpha=alpha)

    if eer_idx is not None:
        x, y = rates['EER'][eer_idx]
        ax.vlines(x, 0, 1, 'r', lw=lw)
        ax.set_title('EER = %0.2f %%' % (100. * y))

    return fig


def plot_biometrics(assessment=None, eer_idx=None, path=None, show=False):
    """Create a summary plot of a biometrics test run.

    Parameters
    ----------
    assessment : dict
        Classification assessment results.
    eer_idx : int, optional
        Classifier reference index for the Equal Error Rate.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure()
    fig.suptitle('Biometrics Summary')

    c_sub = ['#008bff', '#8dd000']
    c_global = ['#0037ff', 'g']

    ths = assessment['thresholds']

    auth_ax = fig.add_subplot(121)
    id_ax = fig.add_subplot(122)

    # subject results
    for sub in six.iterkeys(assessment['subject']):
        auth_rates = assessment['subject'][sub]['authentication']['rates']
        _ = _plot_rates(ths, auth_rates, ['FAR', 'FRR'],
                        lw=MINOR_LW,
                        colors=c_sub,
                        alpha=0.4,
                        eer_idx=None,
                        labels=False,
                        ax=auth_ax)

        id_rates = assessment['subject'][sub]['identification']['rates']
        _ = _plot_rates(ths, id_rates, ['MR', 'RR'],
                        lw=MINOR_LW,
                        colors=c_sub,
                        alpha=0.4,
                        eer_idx=None,
                        labels=False,
                        ax=id_ax)

    # global results
    auth_rates = assessment['global']['authentication']['rates']
    _ = _plot_rates(ths, auth_rates, ['FAR', 'FRR'],
                    lw=MAJOR_LW,
                    colors=c_global,
                    alpha=1,
                    eer_idx=eer_idx,
                    labels=True,
                    ax=auth_ax)

    id_rates = assessment['global']['identification']['rates']
    _ = _plot_rates(ths, id_rates, ['MR', 'RR'],
                    lw=MAJOR_LW,
                    colors=c_global,
                    alpha=1,
                    eer_idx=eer_idx,
                    labels=True,
                    ax=id_ax)

    # set labels and grids
    auth_ax.set_xlabel('Threshold')
    auth_ax.set_ylabel('Authentication')
    auth_ax.grid()
    auth_ax.legend()

    id_ax.set_xlabel('Threshold')
    id_ax.set_ylabel('Identification')
    id_ax.grid()
    id_ax.legend()

    # make layout tight
    fig.tight_layout()

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def plot_clustering(data=None, clusters=None, path=None, show=False):
    """Create a summary plot of a data clustering.

    Parameters
    ----------
    data : array
        An m by n array of m data samples in an n-dimensional space.
    clusters : dict
        Dictionary with the sample indices (rows from `data`) for each cluster.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure()
    fig.suptitle('Clustering Summary')

    ymin, ymax = _yscaling(data, alpha=1.2)

    # determine number of clusters
    keys = list(clusters)
    nc = len(keys)

    if nc <= 4:
        nrows = 2
        ncols = 4
    else:
        area = nc + 4

        # try to fit to a square
        nrows = int(np.ceil(np.sqrt(area)))

        if nrows > MAX_ROWS:
            # prefer to increase number of columns
            nrows = MAX_ROWS

        ncols = int(np.ceil(area / float(nrows)))

    # plot grid
    gs = gridspec.GridSpec(nrows, ncols, hspace=0.2, wspace=0.2)

    # global axes
    ax_global = fig.add_subplot(gs[:2, :2])

    # cluster axes
    c_grid = np.ones((nrows, ncols), dtype='bool')
    c_grid[:2, :2] = False
    c_rows, c_cols = np.nonzero(c_grid)

    # generate color map
    x = np.linspace(0., 1., nc)
    cmap = plt.get_cmap('rainbow')

    for i, k in enumerate(keys):
        aux = data[clusters[k]]
        color = cmap(x[i])
        label = 'Cluster %s' % k
        ax = fig.add_subplot(gs[c_rows[i], c_cols[i]], sharex=ax_global)
        ax.set_ylim([ymin, ymax])
        ax.set_title(label)
        ax.grid()

        if len(aux) > 0:
            ax_global.plot(aux.T, color=color, lw=MINOR_LW, alpha=0.7)
            ax.plot(aux.T, color=color, lw=MAJOR_LW)

    ax_global.set_title('All Clusters')
    ax_global.set_ylim([ymin, ymax])
    ax_global.grid()

    # make layout tight
    gs.tight_layout(fig)

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)
