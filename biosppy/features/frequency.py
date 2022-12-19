# Imports
# 3rd party
import numpy as np
from .. import utils
from . import time

#local
from scipy import interpolate


def getbands(frequencies, fband= [0, 1]):
   band = np.argwhere((frequencies >= fband[0]) & (frequencies <= fband[1])).reshape(-1)
   return frequencies[band]


def freq_features(signal, sampling_rate):
    """Compute spectral metrics describing the signal.
    
        Parameters
        ----------
        signal : array
            Input signal.
        sampling_rate : float
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
        spectral_hist_ : list
            Histogram of the signal spectrum.

        References
        ----------
        TSFEL library: https://github.com/fraunhoferportugal/tsfel
        Peeters, Geoffroy. (2004). A large set of audio features for sound description (similarity and classification) in the CUIDADO project.

        - [0, 0.1], [0.1,0.2] , [0.2,0.3], [0.3, 0.4]: J. Wang and Y. Gong, “Recognition of multiple drivers’s emotional state,” in 2008 19th International Conference on Pattern Recognition, Dec 2008, pp. 1–4.
        - [0.05–5] was split into five bands - power + [0.05–1 Hz] - stat - Ghaderyan, P. and Abbasi, A., 2016. An efficient automatic workload estimation method based on electrodermal activity using pattern classifier combinations. International Journal of Psychophysiology, 110, pp.91-101.
        - temp fts on [0.05−0.50] was split into five bands + stats fts on FFT  - Shukla, Jainendra, et al. "Feature extraction and selection for emotion recognition from electrodermal activity." IEEE Transactions on Affective Computing 12.4 (2019): 857-869
        - FFT for bands (0.1, 0.2), F2 (0.2, 0.3) and F3 (0.3, 0.4) - Sánchez-Reolid, R., de la Rosa, F.L., Sánchez-Reolid, D., López, M.T., Fernández-Caballero, A. (2021). Feature and Time Series Extraction in Artificial Neural Networks for Arousal Detection from Electrodermal Activity. In: Rojas, I., Joya, G., Català, A. (eds) Advances in Computational Intelligence. IWANN 2021. Lecture Notes in Computer Science(), vol 12861. Springer, Cham. 
        """
        
    # check inputs
    assert len(signal) > 0, 'Signal size < 1'

    # ensure numpy
    signal = np.array(signal)
    window = np.hamming(len(signal))
    signal = signal * window

    spectrum = np.abs(np.fft.fft(signal, sampling_rate))
    
    f = np.nan_to_num(np.array(np.fft.fftfreq(len(spectrum))))
    spectrum = np.nan_to_num(spectrum[:len(spectrum)//2])
    spectrum /= len(spectrum)
    f = np.abs(f[:len(f)//2]*sampling_rate)

    args, names = [], []

    # temporal
    _fts = time.time_features(spectrum, sampling_rate)
    fts_name = [str("FFT_" + i) for i in _fts.keys()]
    fts = list(_fts[:])

    args += fts
    names += fts_name

    # fundamental_frequency
    try:
        fundamental_frequency = f[np.argmax(spectrum)]
    except Exception as e:
        print(e)
        fundamental_frequency = None
    args += [fundamental_frequency]
    names += ['fundamental_frequency']

    # harmonic sum
    try:
        if fundamental_frequency > 0:
            harmonics = np.array([n * fundamental_frequency for n in range(2, int((sampling_rate/2)/fundamental_frequency), 1)]).astype(int)
            sp_hrm = spectrum[np.array([np.where(f >= h)[0][0] for h in harmonics])]
            sum_harmonics = np.sum(sp_hrm)
        else:
            sum_harmonics = 0
    except Exception as e:
        print(e)
        sum_harmonics = None
    args += [sum_harmonics]
    names += ['sum_harmonics']

    # spectral_roll_on
    en_sp = spectrum**2#*(f[1]-f[0])
    cum_en = np.cumsum(en_sp)

    try:
        norm_cm_s = cum_en/cum_en[-1]
    except Exception as e:
        print(e)
        norm_cm_s =cum_en
    try:
        spectral_roll_on = f[np.argwhere(norm_cm_s >= 0.05)[0][0]]
    except:
        spectral_roll_on = None

    args += [spectral_roll_on]
    names += ['spectral_roll_on']

    # spectral_roll_off
    try:
        spectral_roll_off = f[np.argwhere(norm_cm_s >= 0.95)[0][0]]
    except:
        spectral_roll_off = None
    args += [spectral_roll_off]
    names += ['spectral_roll_off']

    # histogram
    try:
        _hist = list(np.histogram(spectrum, bins=5)[0])
        _hist = _hist/np.sum(_hist)
    except:
        _hist = [None] * 5

    args += [i for i in _hist]
    names += ['spectral_hist_' + str(i) for i in range(len(_hist))]

    # bands
    spectrum = np.nan_to_num(np.abs(np.fft.fft(signal, sampling_rate*5)))
    f = np.nan_to_num(np.array(np.fft.fftfreq(len(spectrum))))

    spectrum = np.nan_to_num(spectrum[:len(spectrum)//2])
    spectrum /= len(spectrum)
    f = np.abs(f[:len(f)//2]*sampling_rate)

    _f = interpolate.interp1d(f, spectrum)
    resSR = 500
    f = np.arange(f[0], f[-1], 1/resSR)
    spectrum = np.array(_f(f))

    f_b = getbands(f, fband = [0.05, 0.1])
    
    # temporal
    _fts = time.time_features(f_b, resSR)
    fts_name = [str("FFT_005_01" + i) for i in _fts.keys()]
    fts = list(_fts[:])

    args += fts
    names += fts_name

    f_b = getbands(f, fband = [0.1, 0.2])
    # temporal
    _fts = time.time_features(f_b, resSR)
    fts_name = [str("FFT_01_02" + i) for i in _fts.keys()]
    fts = list(_fts[:])

    args += fts
    names += fts_name

    f_b = getbands(f, fband = [0.2, 0.3])
    # temporal
    _fts = time.time_features(f_b, resSR)
    fts_name = [str("FFT_02_03" + i) for i in _fts.keys()]
    fts = list(_fts[:])

    args += fts
    names += fts_name

    f_b = getbands(f, fband = [0.3, 0.4])
    # temporal
    _fts = time.time_features(f_b, resSR)
    fts_name = [str("FFT_03_04" + i) for i in _fts.keys()]
    fts = list(_fts[:])

    args += fts
    names += fts_name

    f_b = getbands(f, fband = [0.4, 0.5])
    # temporal
    _fts = time.time_features(f_b, resSR)
    fts_name = [str("FFT_04_05" + i) for i in _fts.keys()]
    fts = list(_fts[:])

    args += fts
    names += fts_name
    
    args = np.nan_to_num(args)
    return utils.ReturnTuple(tuple(args), tuple(names))
