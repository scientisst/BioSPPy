import numpy as np
from .. import utils
from sklearn import linear_model 
from ..signals import tools
from scipy.stats import iqr, stats, entropy
from ..stats import pearson_correlation


def mob(signal):
    """Compute signal mobility hjorth feature.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    mobility : float
        Signal mobility. 
    """
    args, names = [], []
    d = np.diff(signal)
    args += [np.sqrt(np.var(d)/np.var(signal))]
    names += ['mobility']
    args = np.nan_to_num(args)
    return utils.ReturnTuple(tuple(args), tuple(names))


def com(signal):
    """Compute signal complexity hjorth feature.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    complexity : float
        Signal complexity. 
    """
    args, names = [], []
    d = np.diff(signal)
    args += [mob(d)["mobility"]/mob(signal)["mobility"]]
    names += ['complexity']
    args = np.nan_to_num(args)
    return utils.ReturnTuple(tuple(args), tuple(names))


def chaos(signal):
    """Compute signal chaos hjorth feature.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    chaos : float
        Signal chaos. 
    """
    args, names = [], []
    d = np.diff(signal)
    args += [com(d)['complexity']/com(signal)['complexity']]
    names += ['chaos']
    args = np.nan_to_num(args)
    return utils.ReturnTuple(tuple(args), tuple(names))


def time_features(signal, sampling_rate):
    """Compute various time metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    sampling_rate : float
        Sampling Rate
    Returns
    -------
    max : float
        Signal maximum amplitude.

    min: float
        Signal minimum amplitude.

    range: float
        Signal range amplitude.
    
    iqr : float
        Interquartil range

    mean : float
        Signal average
    
    std : float
        Signal standard deviation
    
    maxToMean: float
        Signal maximum aplitude to mean.
    
    dist : float
        Length of the signal (sum of abs diff)

    meanAD1 : float
        Mean absolute differences.

    medAD1 : float
        Median absolute differences.

    minAD1 : float
        Min absolute differences.

    maxAD1 : float
        Maximum absolute differences.

    meanD1 : float
        Mean of differences.

    medD1 : float
        Median of differences.

    stdD1 : float
        Standard deviation of differences.

    maxD1 : float
        Max of differences.

    minD1 : float
        Min of differences.

    sumD1 : float
        Sum of differences.

    rangeD1 : float
        Amplitude range of differences.

    iqrd1 : float
        interquartil range of differences.

    meanD2 : float
        Mean of 2nd differences.

    stdD2 : float
        Standard deviation of 2nd differences.

    maxD2 : float
        Max of 2nd differences.

    minD2 : float
        Min of 2nd differences.

    sumD2: float
        Sum of 2nd differences.

    rangeD2 : float
        Amplitude range of 2nd differences.

    iqrD2 : float
        Interquartile range of 2nd differences.

    autocorr : float
        Signal autocorrelation sum.

    zeroCross : int
        Number of times the sinal crosses the zero axis.

    CminPks : int
        Number of minimum peaks.

    CmaxPks : int
        Number of maximum peaks.   

    totalE : float
        Total energy.

    linRegSlope : float
        Slope of linear regression. 
    
    linRegb : float
        Interception coefficient b of linear regression. 

    degreLin: float
        Degree of linearity

    mobility: float
        ratio of the variance between the first derivative and the signal

    complexity: float
        ratio between the mobility of the derivative and the mobility of the signal

    chaos: float
        ratio between the complexity of the derivative and the complexity of the signal

    hazard: float
        ratio between the chaos of the derivative and the chaos of the signal

    kurtosis : float
        Signal kurtosis (unbiased).

    skewness : float
        Signal skewness (unbiased).

    rms : float
        Root Mean Square.

    midhinge: float
        average of first and third quartile

    trimean: float
        weighted average of 1st, 2nd and 3rd quartiles

    stat_hist : list
        Histogram.
    
    entropy : float
        Signal entropy.
   
    References
    ----------
    TSFEL library: https://github.com/fraunhoferportugal/tsfel
    Peeters, Geoffroy. (2004). A large set of audio features for sound description (similarity and classification) in the CUIDADO project.
    Veeranki, Yedukondala Rao, Nagarajan Ganapathy, and Ramakrishnan Swaminathan. "Non-Parametric Classifiers Based Emotion Classification Using Electrodermal Activity and Modified Hjorth Features." MIE. 2021.
    Ghaderyan, Peyvand, and Ataollah Abbasi. "An efficient automatic workload estimation method based on electrodermal activity using pattern classifier combinations." International Journal of Psychophysiology 110 (2016): 91-101.

    """

    # check input
    assert len(signal) > 0, 'Signal size < 1'
    # ensure numpy
    signal = np.array(signal)
   
    # helpers
    args, names = [], []
    try:
        sig_diff = np.diff(signal)
    except Exception as e:
        print(e)
        sig_diff = []
    ## 2nd derivative
    try:
        sig_diff_2 = np.diff(sig_diff)
    except Exception as e:
        print(e)
        sig_diff_2 = []

    try:
        mean = np.mean(signal)
    except Exception as e:
        print(e)
        mean = None

    try:
        time = range(len(signal))
        time = [float(x) / sampling_rate for x in time]
    except Exception as e:
        print(e)
        time = []

    try:
        ds = 1/sampling_rate
        energy = np.sum(signal**2*ds)
    except Exception as e:
        print(e)   
        energy = []

    ### end helpers
    
    # signal max
    try:
        _max = np.max(signal)
    except Exception as e:
        print(e)   
        _max = None
    args += [_max]
    names += ['max']
    
    # signal min
    try:
        _min = np.min(signal)
    except Exception as e:
        print(e)   
        _min = None
    args += [_min]
    names += ['min']
    
    # range
    try:
        _range = np.max(signal) - np.min(signal)
    except Exception as e:
        print(e)   
        _range = None
    args += [_range]
    names += ['range']

    # interquartile range 
    try:
        _iqr = iqr(signal)
    except Exception as e:
        print(e)
        _iqr = None
    args += [_iqr]
    names += ['iqr']

    # mean
    try:
        mean = np.mean(signal)
    except Exception as e:
        print(e)   
        mean = None
    args += [mean]
    names += ['mean']

    # std
    try:
        std = np.std(signal) 
    except Exception as e:
        print(e)   
        std = None
    args += [std]
    names += ['std']

    # max to mean
    try:
        maxToMean = np.max(signal - mean)
    except Exception as e:
        print(e)   
        maxToMean = None
    args += [maxToMean]
    names += ['maxToMean']

    # distance
    try:
        dist = np.sum([1 if d == 0 else d for d in np.abs(sig_diff)]) +1
    except Exception as e:
        print(e)   
        dist = None
    args += [dist]
    names += ['dist']

    # mean absolute differences
    try:
        meanAD1 = np.mean(np.abs(sig_diff))
    except Exception as e:
        print(e)   
        meanAD1 = None
    args += [meanAD1]
    names += ['meanAD1']

    # median absolute differences
    try:
        medAD1 = np.median(np.abs(sig_diff))
    except Exception as e:
        print(e)
        medAD1 = None
    args += [medAD1]
    names += ['medAD1']

    # min absolute differences
    try:
        minAD1 = np.min(np.abs(sig_diff))
    except Exception as e:
        print(e)   
        minAD1 = None
    args += [minAD1]
    names += ['minAD1']

    # max absolute differences
    try:
        maxAD1 = np.max(np.abs(sig_diff))
    except Exception as e:
        print(e)   
        maxAD1 = None
    args += [maxAD1]
    names += ['maxAD1']

    # mean of differences
    try:
        meanD1 = np.mean(sig_diff)
    except Exception as e:
        print(e)  
        meanD1 = None
    args += [meanD1]
    names += ['meanD1']

    # median of differences
    try:
        medD1 = np.median(sig_diff)
    except Exception as e:
        print(e)  
        medD1 = None
    args += [medD1]
    names += ['medD1']

    # std of differences
    try:
        stdD1 = np.std(sig_diff)
    except Exception as e:
        print(e)  
        stdD1 = None
    args += [stdD1]
    names += ['stdD1']
    
    # max of differences
    try:
        maxD1 = np.max(sig_diff)
    except Exception as e:
        print(e)  
        maxD1 = None
    args += [maxD1]
    names += ['maxD1']

    # min of differences
    try:
        minD1 = np.min(sig_diff)
    except Exception as e:
        print(e)   
        minD1 = None
    args += [minD1]
    names += ['minD1']

    # sum of differences
    try:
        sumD1 = np.sum(sig_diff)
    except Exception as e:
        print(e)
        sumD1 = None
    args += [sumD1]
    names += ['sumD1']

    # range of differences
    try:
        rangeD1 = np.max(sig_diff) - np.min(sig_diff)
    except Exception as e:
        print(e)
        rangeD1 = None
    args += [rangeD1]
    names += ['rangeD1']

    # interquartile range of differences
    try:
        iqrD1 = iqr(sig_diff)
    except Exception as e:
        print(e)
        iqrD1 = None
    args += [iqrD1]
    names += ['iqrD1']

    # mean of 2nd differences
    try:
        meanD2 = np.mean(sig_diff_2)
    except Exception as e:
        print(e)   
        meanD2 = None
    args += [meanD2]
    names += ['meanD2']

    # std of 2nd differences
    try:
        stdD2 = np.std(sig_diff_2)
    except Exception as e:
        print(e)   
        stdD2 = None
    args += [stdD2]
    names += ['stdD2']
    
    # max of 2nd differences
    try:
        maxD2 = np.max(sig_diff_2)
    except Exception as e:
        print(e)   
        maxD2 = None
    args += [maxD2]
    names += ['maxD2']

    # min of 2nd differences
    try:
        minD2 = np.min(sig_diff_2)
    except Exception as e:
        print(e)   
        minD2 = None
    args += [minD2]
    names += ['minD2']

    # sum of 2nd differences
    try:
        sumD2 = np.sum(sig_diff_2)
    except Exception as e:
        print(e)   
        sumD2 = None
    args += [sumD2]
    names += ['sumD2']

    # range of 2nd differences
    try:
        rangeD2 = np.max(sig_diff_2) - np.min(sig_diff_2)
    except Exception as e:
        print(e)   
        rangeD2 = None
    args += [rangeD2]
    names += ['rangeD2']

    # interquartile range of 2nd differences
    try:
        iqrD2 = iqr(sig_diff_2)
    except Exception as e:
        print(e)
        iqrD2 = None
    args += [iqrD2]
    names += ['iqrD2']

    # autocorrelation sum
    try:
        autocorr = np.sum(np.correlate(signal, signal, 'full'))
    except Exception as e:
        print(e)   
        autocorr = None
    args += [autocorr]
    names += ['autocorr']

    # zero_cross
    try:
        zeroCross = len(np.where(np.abs(np.diff(np.sign(signal))) >= 1)[0])
    except Exception as e:
        print(e)   
        zeroCross = None
    args += [zeroCross]
    names += ['zeroCross']

    # number of minimum peaks
    try:
        CminPks = len(tools.find_extrema(signal, "min")["extrema"])
    except Exception as e:
        print(e)   
        CminPks = None
    args += [CminPks]
    names += ['CminPks']

    # number of maximum peaks
    try:
        CmaxPks = len(tools.find_extrema(signal, "max")["extrema"])
    except Exception as e:
        print(e)   
        CmaxPks = None
    args += [CmaxPks]
    names += ['CmaxPks']
    
    # total energy
    try:
        totalE = np.sum(energy)
    except Exception as e:
        print(e)   
        totalE = None
    args += [totalE]
    names += ['totalE']

    _t = np.array(time).reshape(-1, 1)
    try:
        reg = linear_model.LinearRegression().fit(_t,  signal) 
        linRegSlope = reg.coef_[0]    
    except Exception as e:
        print(e)   
        linRegSlope = None
    args += [linRegSlope]
    names += ['linRegSlope']

    try:
        linRegb = reg.intercept_    
    except Exception as e:
        print(e)   
        linRegb = None
    args += [linRegb]
    names += ['linRegb']

    try:    
        degreeLin = pearson_correlation(signal, reg.predict(_t))[0]
    except Exception as e:
        print(e)  
        degreeLin = None
    args += [degreeLin]
    names += ['degreeLin']
    

    ## hjorth
    # mobility
    try:    
        mobility = mob(signal)['mobility']
    except Exception as e:
        print(e)  
        mobility = None
    args += [mobility]
    names += ['mobility']

    # complexity
    try:    
        complexity = com(signal)['complexity']
    except Exception as e:
        print(e)  
        complexity = None
    args += [complexity]
    names += ['complexity']

    # chaos
    try:    
        _chaos = chaos(signal)['chaos']
    except Exception as e:
        print(e)  
        _chaos = None
    args += [_chaos]
    names += ['chaos']

    # Hazard
    try:    
        hazard = chaos(sig_diff)['chaos']/chaos(signal)['chaos']
    except Exception as e:
        print(e)  
        hazard = None
    args += [hazard]
    names += ['hazard']

    # kurtosis
    try:
        kurtosis = stats.kurtosis(signal, bias=False)
    except Exception as e:
        print(e) 
        kurtosis = None
    args += [kurtosis]
    names += ['kurtosis']

    # skweness
    try:
        skewness = stats.skew(signal, bias=False)
    except Exception as e:
        print(e) 
        skewness = None
    args += [skewness]
    names += ['skewness']

    # root mean square
    try:
        rms = np.sqrt(np.sum(signal ** 2) / len(signal))
    except Exception as e:
        print(e) 
        rms = None
    args += [rms]
    names += ['rms']

    # midhinge
    try:
        quant = np.quantile(signal, [0.25, 0.5, 0.75])
        midhinge = (quant[0] + quant[2])/2
    except Exception as e:
        print(e) 
        midhinge = None
    args += [midhinge]
    names += ['midhinge']

    # trimean
    try:
        trimean = (quant[1] + midhinge)/2
    except Exception as e:
        print(e) 
        trimean = None
    args += [trimean]
    names += ['trimean']

    # histogram
    try:
        _hist = list(np.histogram(signal, bins=5)[0])
        _hist = _hist/np.sum(_hist)
    except Exception as e:
        print(e) 
        _hist = [None] * 5
    args += [i for i in _hist]
    names += ['stat_hist' + str(i) for i in range(len(_hist))]

    # entropy
    try:
        _entropy = entropy(signal)
    except Exception as e:
        print("entropy", e) 
        _entropy = None
    args += [_entropy]
    names += ['entropy']

    # output
    args = np.nan_to_num(args)
    return utils.ReturnTuple(tuple(args), tuple(names))
