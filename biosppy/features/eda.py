import numpy as np
from .. import utils
from .. import tools as st


def eda_param(signal, min_amplitude=0.08, filt=True, size=1., sampling_rate= 1000.):
    """ Returns characteristic EDA events.

    Parameters
    ----------
    signal : array
        Input signal.
    min_amplitude : float, optional
        Minimum threshold by which to exclude SCRs.
    filt: bool
        If to filter signal to remove noise and low amplitude events.
    size: float
        Size of the filter in seconds
    sampling_rate: float
        Data acquisition sampling rate.

    Returns
    -------
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.
    amplitudes : array
        SCR pulse amplitudes.
    end : array
        Indices of the SCR end.
    """
    
    assert len(signal) > 1, "len signal <1"
    signal = np.array(signal).astype(np.float)

    # smooth
    if filt:
        try:
            if sampling_rate > 1:
                signal, _, _ = st.filter_signal(signal=signal,
                                         ftype='butter',
                                         band='lowpass',
                                         order=4,
                                         frequency=2,
                                         sampling_rate=sampling_rate)
        except Exception as e:
            print(e, "Error filtering EDA")

        # smooth
        try:
            sm_size = int(size * sampling_rate)
            signal, _ = st.smoother(signal=signal,
                                      kernel='boxzen',
                                      size=sm_size,
                                      mirror=True)
        except Exception as e: print(e)


    amps, onsets, peaks, end = [], [], [], []
    zeros = st.find_extrema(signal=signal, mode='min')[0]  # get zeros
    for z in range(len(zeros)):
        if z == len(zeros) -1:  # last zero
            s = signal[zeros[z]:]  # signal amplitude between event
        else:
            s = signal[zeros[z]:zeros[z + 1]]  # signal amplitude between event
            
        pk = st.find_extrema(signal=s, mode='max')[0]  # get pk between events
        for p in pk:
            if (s[p] - s[0]) > min_amplitude:  # only count events with high amplitude
                peaks += [zeros[z] + p]
                onsets += [zeros[z]]
                amps += [s[p] - s[0]]
                if z == len(zeros) -1:  # last zero
                    end += [len(signal)]                    
                else:
                    end += [zeros[z + 1]]

    args, names = [], []
    names += ["onsets", "peaks", "amplitudes", "end"]
    args += [onsets, peaks, amps, end]

    args = np.nan_to_num(args)
    return utils.ReturnTuple(tuple(args), tuple(names))


def rec_times(signal, onsets, peaks):
    """ Returns EDA recovery times.

    Parameters
    ----------
    signal : array
        Input signal.
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.

    Returns
    -------
    half_rec : list
        Half Recovery times, i.e. time between peak and 50% amplitude.

    six_rec : list
        63 % recovery times, i.e. time between peak and 63% amplitude.

    """
    assert len(signal) > 1, "len signal <1"
    peaks = np.array(peaks).astype(int) 
    onsets = np.array(onsets).astype(int) 

    a = np.array(signal[peaks[:]] - signal[onsets[:]])
    li = min(len(onsets), len(peaks))

    half_rec, hlf_rec_ts = [], []
    six_rec, six_rec_ts = [], []
    for i in range(li):  # iterate over onset
        half_rec_amp = 0.5 * a[i] + signal[onsets][i]
        six_rec_amp = 0.37 * a[i] + signal[onsets][i]
        try:
            wind = np.array(signal[peaks[i]:onsets[i + 1]])
        except:
            wind = np.array(signal[peaks[i]:])
        half_rec_idx = np.argwhere(wind <= half_rec_amp)
        six_rec_idx = np.argwhere(wind <= six_rec_amp)
        
        if len(half_rec_idx) > 0:
            half_rec += [half_rec_idx[0][0] + peaks[i] - onsets[i]]
            hlf_rec_ts += [half_rec_idx[0][0] + peaks[i]]
        else:
            half_rec += [None]
            hlf_rec_ts += [None]

        if len(six_rec_idx) > 0:
            six_rec += [six_rec_idx[0][0] + peaks[i] - onsets[i]]
            six_rec_ts += [six_rec_idx[0][0] + pks[i]]            
        else:
            six_rec += [None]
            six_rec_ts += [None]

    args, names = [], []
    names += ["half_rec", "six_rec"]
    args += [half_rec, six_rec]
    args = np.nan_to_num(args)
    return utils.ReturnTuple(tuple(args), tuple(names))


def eda_features(signal=None, min_amplitude=0.08, filt=True, size= 1.5, sampling_rate=1000.):
    """Compute EDA characteristic metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.
    min_amplitude : float, optional
        Minimum treshold by which to exclude SCRs.
    filt: bool
        If to filter signal to remove noise and low amplitude events.
    size: float
        Size of the filter in seconds
    sampling_rate : float
        Sampling frequency.
    Returns
    -------
    onsets : list
        Signal EDR events onsets.

    peaks : list
        Signal EDR events peaks.

    amps : list
        Signal EDR events Amplitudes.

    phasic_rate : list
        Signal EDR events rate in 60s.

    rise_ts : list
        Rise times, i.e. onset-peak time difference.

    half_rec : list
        Half Recovery times, i.e. time between peak and 63% amplitude.

    six_rec : list
        63 % recovery times, i.e. time between peak and 50% amplitude.

    """
    # ensure numpy
    assert len(signal) > 0, "len signal < 1"

    signal = np.array(signal)
    args, names = [], []

    # onsets, peaks, amps
    onsets, peaks, amps, _ = eda_param(signal, filt=filt, size=size, min_amplitude=min_amplitude, sampling_rate=sampling_rate)
    args += [onsets]
    names += ['onsets']
    args += [peaks]
    names += ['peaks']
    args += [amps]
    names += ['amps']

    # phasic_rate
    try:
        phasic_rate = sampling_rate * (60. / np.diff(peaks))
    except:
        phasic_rate = None
    args += [phasic_rate]
    names += ['phasic_rate']

    # rise_ts
    try:
        rise_ts = peaks - onsets
    except:
        rise_ts = None
    args += [rise_ts]
    names += ['rise_ts']

    half_rec, six_rec = rec_times(signal, onsets, pks)
        
    args += [half_rec]
    names += ['half_rec']

    args += [six_rec]
    names += ['six_rec']

    args = np.nan_to_num(args)

    return utils.ReturnTuple(tuple(args), tuple(names))
