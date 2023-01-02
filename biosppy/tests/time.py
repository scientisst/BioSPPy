import numpy as np
from ..features.time import time_features


def getData(LEN=100, sampling_rate=100):
    const_0 = np.zeros(LEN)
    const_1 = np.ones(LEN)
    const_neg = -1 * np.ones(LEN)

    f = 5
    x = np.arange(0, LEN/sampling_rate, 1/sampling_rate)
    sine = np.sin(2 * np.pi * f * x)
    np.random.seed(0)
    lin = np.arange(LEN)

    return const_0, const_1, const_eg, lin, sine


def test(LEN=100, sampling_rate=100):
    const0, const1, constNeg, lin, sine = getData(LEN)
    
    const_0_fts = time_features(const_0, sampling_rate)
    const_1_fts = time_features(const_1, sampling_rate)
    const_neg_fts = time_features(const_neg, sampling_rate)
    lin_fts = time_features(lin, sampling_rate)
    sine_fts = time_features(sine, sampling_rate)

    # max
    np.testing.assert_almost_equal(const_0_fts["max"], 0.0, err_msg="const0 max")
    np.testing.assert_almost_equal(const_1_fts["max"], 1.0, err_msg="const1 max")
    np.testing.assert_almost_equal(const_neg_fts["max"], -1.0, err_msg="const neg max")
    np.testing.assert_almost_equal(lin_fts["max"], 99.0, err_msg="lin max")
    np.testing.assert_almost_equal(sine_fts["max"], 1.0, err_msg="sine max")

    # min
    np.testing.assert_almost_equal(const_0_fts["min"], 0.0, err_msg="const0 min")
    np.testing.assert_almost_equal(const_1_fts["min"], 1.0, err_msg="const1 min")
    np.testing.assert_almost_equal(const_neg_fts["min"], -1.0, err_msg="const neg min")
    np.testing.assert_almost_equal(lin_fts["min"], 0.0, err_msg="lin min")
    np.testing.assert_almost_equal(sine_fts["min"], -1.0, err_msg="sine min")

    # range
    np.testing.assert_almost_equal(const_0_fts["range"], 0.0, err_msg="const0 range")
    np.testing.assert_almost_equal(const_1_fts["range"], 0.0, err_msg="const1 range")
    np.testing.assert_almost_equal(const_neg_fts["range"], 0.0, err_msg="const neg range")
    np.testing.assert_almost_equal(lin_fts["range"], 99.0, err_msg="lin range")
    np.testing.assert_almost_equal(sine_fts["range"], 2.0, err_msg="sine range")

    # iqr
    np.testing.assert_almost_equal(const_0_fts["iqr"], 0.0, err_msg="const0 iqr")
    np.testing.assert_almost_equal(const_1_fts["iqr"], 0.0, err_msg="const1 iqr")
    np.testing.assert_almost_equal(const_neg_fts["iqr"], 0.0, err_msg="const neg iqr")
    np.testing.assert_almost_equal(lin_fts["iqr"], 49.5, err_msg="lin iqr")
    np.testing.assert_almost_equal(sine_fts["iqr"], 1.286, err_msg="sine iqr", decimal=2)

    # mean
    np.testing.assert_almost_equal(const_0_fts["mean"], 0.0, err_msg="const0 mean")
    np.testing.assert_almost_equal(const_1_fts["mean"], 1.0, err_msg="const1 mean")
    np.testing.assert_almost_equal(const_neg_fts["mean"], -1.0, err_msg="const neg mean")
    np.testing.assert_almost_equal(lin_fts["mean"], 49.5, err_msg="lin mean")
    np.testing.assert_almost_equal(sine_fts["mean"], 0.0, err_msg="sine maean")

    # std
    np.testing.assert_almost_equal(const_0_fts["std"], 0.0, err_msg="const0 std")
    np.testing.assert_almost_equal(const_1_fts["std"], 0.0, err_msg="const1 std")
    np.testing.assert_almost_equal(const_neg_fts["std"], 0.0, err_msg="const neg std")
    np.testing.assert_almost_equal(lin_fts["std"], 28.86, err_msg="lin std", decimal=2)
    np.testing.assert_almost_equal(sine_fts["std"], 0.70, err_msg="sine std", decimal=2)
    
    # maxToMean
    np.testing.assert_almost_equal(const_0_fts["maxToMean"], 0.0, err_msg="const0 maxToMean")
    np.testing.assert_almost_equal(const_1_fts["maxToMean"], 0.0, err_msg="const1 maxToMean")
    np.testing.assert_almost_equal(const_neg_fts["maxToMean"], 0.0, err_msg="const neg maxToMean")
    np.testing.assert_almost_equal(lin_fts["maxToMean"], 49.5, err_msg="lin maxToMean")
    np.testing.assert_almost_equal(sine_fts["maxToMean"], 1.0, err_msg="sine maxToMean")

    # dist
    np.testing.assert_almost_equal(const_0_fts["dist"], 100.0, err_msg="const0 dist")
    np.testing.assert_almost_equal(const_1_fts["dist"], 100.0, err_msg="const1 dist")
    np.testing.assert_almost_equal(const_neg_fts["dist"], 100.0, err_msg="const neg dist")
    np.testing.assert_almost_equal(lin_fts["dist"], 100, err_msg="lin dist")
    np.testing.assert_almost_equal(sine_fts["dist"], 20.69, err_msg="sine dist", decimal=2)

    # meanAD
    np.testing.assert_almost_equal(const_0_fts["meanAD1"], 0.0, err_msg="const0 meanAD1")
    np.testing.assert_almost_equal(const_1_fts["meanAD1"], 0.0, err_msg="const1 meanAD1")
    np.testing.assert_almost_equal(const_neg_fts["meanAD1"], 0.0, err_msg="const neg meanAD1")
    np.testing.assert_almost_equal(lin_fts["meanAD1"], 1.0, err_msg="lin meanAD1")
    np.testing.assert_almost_equal(sine_fts["meanAD1"], 0.19, err_msg="sine meanAD1", decimal=2)

    # medAD1
    np.testing.assert_almost_equal(const_0_fts["medAD1"], 0.0, err_msg="const0 medAD1")
    np.testing.assert_almost_equal(const_1_fts["medAD1"], 0.0, err_msg="const1 medAD1")
    np.testing.assert_almost_equal(const_neg_fts["medAD1"], 0.0, err_msg="const neg medAD1")
    np.testing.assert_almost_equal(lin_fts["medAD1"], 1.0, err_msg="lin medAD1")
    np.testing.assert_almost_equal(sine_fts["medAD1"], 0.22, err_msg="sine medAD1", decimal=2)

    # minAD1
    np.testing.assert_almost_equal(const_0_fts["minAD1"], 0.0, err_msg="const0 minAD1")
    np.testing.assert_almost_equal(const_1_fts["minAD1"], 0.0, err_msg="const1 minAD1")
    np.testing.assert_almost_equal(const_neg_fts["minAD1"], 0.0, err_msg="const neg minAD1")
    np.testing.assert_almost_equal(lin_fts["minAD1"], 1.0, err_msg="lin minAD1")
    np.testing.assert_almost_equal(sine_fts["minAD1"], 0.04, err_msg="sine minAD1", decimal=2)

    # maxAD1
    np.testing.assert_almost_equal(const_0_fts["maxAD1"], 0.0, err_msg="const0 maxAD1")
    np.testing.assert_almost_equal(const_1_fts["maxAD1"], 0.0, err_msg="const1 maxAD1")
    np.testing.assert_almost_equal(const_neg_fts["maxAD1"], 0.0, err_msg="const neg maxAD1")
    np.testing.assert_almost_equal(lin_fts["maxAD1"], 1.0, err_msg="lin maxAD1")
    np.testing.assert_almost_equal(sine_fts["maxAD1"], 0.30, err_msg="sine maxAD1", decimal=2)
 
    # meanD1
    np.testing.assert_almost_equal(const_0_fts["meanD1"], 0.0, err_msg="const0 meanD1")
    np.testing.assert_almost_equal(const_1_fts["meanD1"], 0.0, err_msg="const1 meanD1")
    np.testing.assert_almost_equal(const_neg_fts["meanD1"], 0.0, err_msg="const neg meanD1")
    np.testing.assert_almost_equal(lin_fts["meanD1"], 1.0, err_msg="lin meanD1")
    np.testing.assert_almost_equal(sine_fts["meanD1"], -0.003, err_msg="sine meanD1", decimal=2)

    # medD1
    np.testing.assert_almost_equal(const_0_fts["medD1"], 0.0, err_msg="const0 medD1")
    np.testing.assert_almost_equal(const_1_fts["medD1"], 0.0, err_msg="const1 medD1")
    np.testing.assert_almost_equal(const_neg_fts["medD1"], 0.0, err_msg="const neg medD1")
    np.testing.assert_almost_equal(lin_fts["medD1"], 1.0, err_msg="lin medD1")
    np.testing.assert_almost_equal(sine_fts["medD1"], -0.04, err_msg="sine medD1", decimal=2)

    # stdD1
    np.testing.assert_almost_equal(const_0_fts["stdD1"], 0.0, err_msg="const0 stdd1")
    np.testing.assert_almost_equal(const_1_fts["stdD1"], 0.0, err_msg="const1 stdD1")
    np.testing.assert_almost_equal(const_neg_fts["stdD1"], 0.0, err_msg="const neg stdD1")
    np.testing.assert_almost_equal(lin_fts["stdD1"], 0.0, err_msg="lin stdD1", decimal=2)
    np.testing.assert_almost_equal(sine_fts["stdD1"], 0.22, err_msg="sine stdD1", decimal=2)
    
    # minD1
    np.testing.assert_almost_equal(const_0_fts["minD1"], 0.0, err_msg="const0 minD1")
    np.testing.assert_almost_equal(const_1_fts["minD1"], 0.0, err_msg="const1 minD1")
    np.testing.assert_almost_equal(const_neg_fts["minD1"], 0.0, err_msg="const neg minD1")
    np.testing.assert_almost_equal(lin_fts["minD1"], 1.0, err_msg="lin minD1")
    np.testing.assert_almost_equal(sine_fts["minD1"], -0.30, err_msg="sine minD1", decimal=2)

    # maxD1
    np.testing.assert_almost_equal(const_0_fts["maxD1"], 0.0, err_msg="const0 maxD1")
    np.testing.assert_almost_equal(const_1_fts["maxD1"], 0.0, err_msg="const1 maxD1")
    np.testing.assert_almost_equal(const_neg_fts["maxD1"], 0.0, err_msg="const neg maxD1")
    np.testing.assert_almost_equal(lin_fts["maxD1"], 1.0, err_msg="lin maxD1")
    np.testing.assert_almost_equal(sine_fts["maxD1"], 0.30, err_msg="sine maxD1", decimal=2)

    # sumD1
    np.testing.assert_almost_equal(const_0_fts["sumD1"], 0.0, err_msg="const0 sumD1")
    np.testing.assert_almost_equal(const_1_fts["sumD1"], 0.0, err_msg="const1 sumD1")
    np.testing.assert_almost_equal(const_neg_fts["sumD1"], 0.0, err_msg="const neg sumD1")
    np.testing.assert_almost_equal(lin_fts["sumD1"], 99.0, err_msg="lin sumD1")
    np.testing.assert_almost_equal(sine_fts["sumD1"], -0.30, err_msg="sine sumD1", decimal=2)

    # rangeD1
    np.testing.assert_almost_equal(const_0_fts["rangeD1"], 0.0, err_msg="const0 rangeD1")
    np.testing.assert_almost_equal(const_1_fts["rangeD1"], 0.0, err_msg="const1 rangeD1")
    np.testing.assert_almost_equal(const_neg_fts["rangeD1"], 0.0, err_msg="const neg rangeD1")
    np.testing.assert_almost_equal(lin_fts["rangeD1"], 0.0, err_msg="lin rangeD1")
    np.testing.assert_almost_equal(sine_fts["rangeD1"], 0.61, err_msg="sine rangeD1", decimal=2)

    # iqrD1
    np.testing.assert_almost_equal(const_0_fts["iqrD1"], 0.0, err_msg="const0 iqrD1")
    np.testing.assert_almost_equal(const_1_fts["iqrD1"], 0.0, err_msg="const1 iqrD1")
    np.testing.assert_almost_equal(const_neg_fts["iqrD1"], 0.0, err_msg="const neg iqrD1")
    np.testing.assert_almost_equal(lin_fts["iqrD1"], 0.0, err_msg="lin iqrD1")
    np.testing.assert_almost_equal(sine_fts["iqrD1"], 0.44, err_msg="sine iqrD1", decimal=2)
    
    # meanD2
    np.testing.assert_almost_equal(const_0_fts["meanD2"], 0.0, err_msg="const0 meanD2")
    np.testing.assert_almost_equal(const_1_fts["meanD2"], 0.0, err_msg="const1 meanD2")
    np.testing.assert_almost_equal(const_neg_fts["meanD2"], 0.0, err_msg="const neg meanD2")
    np.testing.assert_almost_equal(lin_fts["meanD2"], 0.0, err_msg="lin meanD2")
    np.testing.assert_almost_equal(sine_fts["meanD2"], -0.003, err_msg="sine meanD2", decimal=2)

    # stdD2
    np.testing.assert_almost_equal(const_0_fts["stdD2"], 0.0, err_msg="const0 stdd2")
    np.testing.assert_almost_equal(const_1_fts["stdD2"], 0.0, err_msg="const1 stdD2")
    np.testing.assert_almost_equal(const_neg_fts["stdD2"], 0.0, err_msg="const neg stdD2")
    np.testing.assert_almost_equal(lin_fts["stdD2"], 0.0, err_msg="lin stdD2", decimal=2)
    np.testing.assert_almost_equal(sine_fts["stdD2"], 0.06, err_msg="sine stdD2", decimal=2)
    
    # minD2
    np.testing.assert_almost_equal(const_0_fts["minD2"], 0.0, err_msg="const0 minD2")
    np.testing.assert_almost_equal(const_1_fts["minD2"], 0.0, err_msg="const1 minD2")
    np.testing.assert_almost_equal(const_neg_fts["minD2"], 0.0, err_msg="const neg minD2")
    np.testing.assert_almost_equal(lin_fts["minD2"], 0.0, err_msg="lin minD2")
    np.testing.assert_almost_equal(sine_fts["minD2"], -0.09, err_msg="sine minD2", decimal=2)

    # maxD2
    np.testing.assert_almost_equal(const_0_fts["maxD2"], 0.0, err_msg="const0 maxD2")
    np.testing.assert_almost_equal(const_1_fts["maxD2"], 0.0, err_msg="const1 maxD2")
    np.testing.assert_almost_equal(const_neg_fts["maxD2"], 0.0, err_msg="const neg maxDr")
    np.testing.assert_almost_equal(lin_fts["maxD2"], 0.0, err_msg="lin maxD2")
    np.testing.assert_almost_equal(sine_fts["maxD2"], 0.09, err_msg="sine maxD2", decimal=2)

    # sumD2
    np.testing.assert_almost_equal(const_0_fts["sumD2"], 0.0, err_msg="const0 sumD2")
    np.testing.assert_almost_equal(const_1_fts["sumD2"], 0.0, err_msg="const1 sumD2")
    np.testing.assert_almost_equal(const_neg_fts["sumD2"], 0.0, err_msg="const neg sumD2")
    np.testing.assert_almost_equal(lin_fts["sumD2"], 0.0, err_msg="lin sumD2")
    np.testing.assert_almost_equal(sine_fts["sumD2"], -0.03, err_msg="sine sumD2", decimal=2)

    # rangeD2
    np.testing.assert_almost_equal(const_0_fts["rangeD2"], 0.0, err_msg="const0 rangeD2")
    np.testing.assert_almost_equal(const_1_fts["rangeD2"], 0.0, err_msg="const1 rangeD2")
    np.testing.assert_almost_equal(const_neg_fts["rangeD2"], 0.0, err_msg="const neg rangeD2")
    np.testing.assert_almost_equal(lin_fts["rangeD2"], 0.0, err_msg="lin rangeD2")
    np.testing.assert_almost_equal(sine_fts["rangeD2"], 0.195, err_msg="sine rangeD2", decimal=2)

    ## iqrD2
    np.testing.assert_almost_equal(const_0_fts["iqrD2"], 0.0, err_msg="const0 iqrD2")
    np.testing.assert_almost_equal(const_1_fts["iqrD2"], 0.0, err_msg="const1 iqrD2")
    np.testing.assert_almost_equal(const_neg_fts["iqrD2"], 0.0, err_msg="const neg iqrD2")
    np.testing.assert_almost_equal(lin_fts["iqrD2"], 0.0, err_msg="lin iqrD2")
    np.testing.assert_almost_equal(sine_fts["iqrD2"], 0.147, err_msg="sine iqrD2", decimal=2)

    ## autocorr
    np.testing.assert_almost_equal(const_0_fts["autocorr"], 0.0, err_msg="const0 autocorr")
    np.testing.assert_almost_equal(const_1_fts["autocorr"], 10000.0, err_msg="const1 autocorr")
    np.testing.assert_almost_equal(const_neg_fts["autocorr"], 10000.0, err_msg="const neg autocorr")
    np.testing.assert_almost_equal(lin_fts["autocorr"], 24502500.0, err_msg="lin autocorr")
    np.testing.assert_almost_equal(sine_fts["autocorr"], 0.0, err_msg="sine autocorr")

    # zeroCross
    np.testing.assert_almost_equal(const_0_fts["zeroCross"], 0.0, err_msg="const0 zeroCross")
    np.testing.assert_almost_equal(const_1_fts["zeroCross"], 0.0, err_msg="const1 zeroCross")
    np.testing.assert_almost_equal(const_neg_fts["zeroCross"], 0.0, err_msg="const neg zeroCross")
    np.testing.assert_almost_equal(lin_fts["zeroCross"], 1.0, err_msg="lin zeroCross")
    np.testing.assert_almost_equal(sine_fts["zeroCross"], 10.0, err_msg="sine zeroCross")

    # CminPks
    np.testing.assert_almost_equal(const_0_fts["CminPks"], 0.0, err_msg="const0 CminPks")
    np.testing.assert_almost_equal(const_1_fts["CminPks"], 0.0, err_msg="const1 CminPks")
    np.testing.assert_almost_equal(const_neg_fts["CminPks"], 0.0, err_msg="const neg CminPks")
    np.testing.assert_almost_equal(lin_fts["CminPks"], 0.0, err_msg="lin CminPks")
    np.testing.assert_almost_equal(sine_fts["CminPks"], 5.0, err_msg="sine CminPks")

    # CmaxPks
    np.testing.assert_almost_equal(const_0_fts["CmaxPks"], 0.0, err_msg="const0 CmaxPks")
    np.testing.assert_almost_equal(const_1_fts["CmaxPks"], 0.0, err_msg="const1 CmaxPks")
    np.testing.assert_almost_equal(const_neg_fts["CmaxPks"], 0.0, err_msg="const neg CmaxPks")
    np.testing.assert_almost_equal(lin_fts["CmaxPks"], 0.0, err_msg="lin CmaxPks")
    np.testing.assert_almost_equal(sine_fts["CmaxPks"], 5.0, err_msg="sine CmaxPks")

    # totalE
    np.testing.assert_almost_equal(const_0_fts["totalE"], 0.0, err_msg="const0 totalE", decimal=2)
    np.testing.assert_almost_equal(const_1_fts["totalE"], 0.99, err_msg="const1 totalE", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts["totalE"], 0.99, err_msg="const neg totalE", decimal=2)
    np.testing.assert_almost_equal(lin_fts["totalE"], 3283.50, err_msg="lin totalE", decimal=2)
    np.testing.assert_almost_equal(sine_fts["totalE"], 0.50, err_msg="sine totalE", decimal=2)

    # linRegSlope
    np.testing.assert_almost_equal(const0_fts["linRegSlope"], 0.0, err_msg="const0 linRegSlope")
    np.testing.assert_almost_equal(const1_fts["linRegSlope"], 0.0, err_msg="const1 linRegSlope")
    np.testing.assert_almost_equal(const_neg_fts["linRegSlope"], 0.0, err_msg="const neg linRegSlope")
    np.testing.assert_almost_equal(lin_fts["linRegSlope"], 100.0, err_msg="lin linRegSlope")
    np.testing.assert_almost_equal(sine_fts["linRegSlope"], -0.378, err_msg="sine linRegSlope", decimal=2)

    # linRegb
    np.testing.assert_almost_equal(const_0_fts["linRegb"], 0.0, err_msg="const0 linRegb")
    np.testing.assert_almost_equal(const_1_fts["linRegb"], 1.0, err_msg="const1 linRegb")
    np.testing.assert_almost_equal(const_neg_fts["linRegb"], -1.0, err_msg="const neg linRegb")
    np.testing.assert_almost_equal(lin_fts["linRegb"], 0.0, err_msg="lin linRegb")
    np.testing.assert_almost_equal(sine_fts["linRegb"], 0.1875, err_msg="sine linRegb", decimal=2)

    # degreeLin
    np.testing.assert_almost_equal(const_0_fts["degreeLin"], 0.0, err_msg="const0 degreeLin")
    np.testing.assert_almost_equal(const_1_fts["degreeLin"], 0.0, err_msg="const1 degreeLin")
    np.testing.assert_almost_equal(const_neg_fts["degreeLin"], 0.0, err_msg="const neg degreeLin")
    np.testing.assert_almost_equal(lin_fts["degreeLin"], 1.0, err_msg="lin degreeLin")
    np.testing.assert_almost_equal(sine_fts["degreeLin"], 0.154, err_msg="sine degreeLin", decimal=2)

    # mobility
    np.testing.assert_almost_equal(const_0_fts["mobility"], 0.0, err_msg="const0 mobility")
    np.testing.assert_almost_equal(const_1_fts["mobility"], 0.0, err_msg="const1 mobility")
    np.testing.assert_almost_equal(const_neg_fts["mobility"], 0.0, err_msg="const neg mobility")
    np.testing.assert_almost_equal(lin_fts["mobility"], 0.0, err_msg="lin mobility", decimal=2)
    np.testing.assert_almost_equal(sine_fts["mobility"], 0.3113312, err_msg="sine mobility", decimal=2)

    # complexity
    np.testing.assert_almost_equal(const_0_fts["complexity"], 0.0, err_msg="const0 complexity")
    np.testing.assert_almost_equal(const_1_fts["complexity"], 0.0, err_msg="const1 complexity")
    np.testing.assert_almost_equal(const_neg_fts["complexity"], 0.0, err_msg="const neg complexity")
    np.testing.assert_almost_equal(lin_fts["complexity"], 0.0, err_msg="lin complexity", decimal=2)
    np.testing.assert_almost_equal(sine_fts["complexity"], 1.019, err_msg="sine complexity", decimal=2)

    # chaos
    np.testing.assert_almost_equal(const_0_fts["chaos"], 0.0, err_msg="const0 chaos")
    np.testing.assert_almost_equal(const_1_fts["chaos"], 0.0, err_msg="const1 chaos")
    np.testing.assert_almost_equal(const_neg_fts["chaos"], 0.0, err_msg="const neg chaos")
    np.testing.assert_almost_equal(lin_fts["chaos"], 0.0, err_msg="lin chaos", decimal=2)
    np.testing.assert_almost_equal(sine_fts["chaos"], 0.9454, err_msg="sine chaos", decimal=2)

    # hazard
    np.testing.assert_almost_equal(const_0_fts["hazard"], 0.0, err_msg="const0 hazard")
    np.testing.assert_almost_equal(const_1_fts["hazard"], 0.0, err_msg="const1 hazard")
    np.testing.assert_almost_equal(const_neg_fts["hazard"], 0.0, err_msg="const neg hazard")
    np.testing.assert_almost_equal(lin_fts["hazard"], 0.0, err_msg="lin hazard", decimal=2)
    np.testing.assert_almost_equal(sine_fts["hazard"], 1.156, err_msg="sine hazard", decimal=2)

    # kurtosis
    np.testing.assert_almost_equal(const_0_fts["kurtosis"],  -3.0, err_msg="const0 kurtosis")
    np.testing.assert_almost_equal(const_1_fts["kurtosis"], -3.0, err_msg="const1 kurtosis")
    np.testing.assert_almost_equal(const_neg_fts["kurtosis"], -3.0, err_msg="const neg kurtosis")
    np.testing.assert_almost_equal(lin_fts["kurtosis"], -1.199, err_msg="lin kurtosis", decimal=2)
    np.testing.assert_almost_equal(sine_fts["kurtosis"], -1.515, err_msg="sine kurtosis", decimal=2)
    tst_fts = time_features([0, 0, 0, 0, 4,5,4,0,0,0,0], sampling_rate=1)
    np.testing.assert_almost_equal(tst_fts["kurtosis"], -0.4198, err_msg="tst kurtosis", decimal=2)

    # skweness
    np.testing.assert_almost_equal(const_0_fts["skewness"], 0.0, err_msg="const0 skewness")
    np.testing.assert_almost_equal(const_1_fts["skewness"], 0.0, err_msg="const1 skewness")
    np.testing.assert_almost_equal(const_neg_fts["skewness"], 0.0, err_msg="const neg skewness")
    np.testing.assert_almost_equal(lin_fts["skewness"], 0.0, err_msg="lin skewness", decimal=2)
    np.testing.assert_almost_equal(sine_fts["skewness"], 0.0, err_msg="sine skewness", decimal=2)

    tst_fts = time_features([4,5,4,0,0,0,0,0,0], sampling_rate=1)
    np.testing.assert_almost_equal(tst_fts["skewness"], 0.9271, err_msg="tst skewness", decimal=2)
    tst_fts = time_features([00,0,0,0,0,0,0,0, 4,5,4], sampling_rate=1)
    np.testing.assert_almost_equal(tst_fts["skewness"], 1.254, err_msg="tst skewness", decimal=2)

    # rms
    np.testing.assert_almost_equal(const_0_fts["rms"], 0.0, err_msg="const0 rms")
    np.testing.assert_almost_equal(const_1_fts["rms"], 1.0, err_msg="const1 rms")
    np.testing.assert_almost_equal(const_neg_fts["rms"], 1.0, err_msg="const neg rms")
    np.testing.assert_almost_equal(lin_fts["rms"], 57.30, err_msg="lin rms", decimal=2)
    np.testing.assert_almost_equal(sine_fts["rms"], 0.707, err_msg="sine rms", decimal=2)

    # midhinge
    np.testing.assert_almost_equal(const_0_fts["midhinge"], 0.0, err_msg="const0 midhinge")
    np.testing.assert_almost_equal(const_1_fts["midhinge"], 1.0, err_msg="const1 midhinge")
    np.testing.assert_almost_equal(const_neg_fts["midhinge"], -1.0, err_msg="const neg midhinge")
    np.testing.assert_almost_equal(lin_fts["midhinge"], 49.5, err_msg="lin midhinge", decimal=2)
    np.testing.assert_almost_equal(sine_fts["midhinge"], 0.0, err_msg="sine midhinge", decimal=2)
   
    # trimean
    np.testing.assert_almost_equal(const_0_fts["trimean"], 0.0, err_msg="const0 trimean")
    np.testing.assert_almost_equal(const_1_fts["trimean"], 1.0, err_msg="const1 trimean")
    np.testing.assert_almost_equal(const_neg_fts["trimean"], -1.0, err_msg="const neg trimean")
    np.testing.assert_almost_equal(lin_fts["trimean"], 49.5, err_msg="lin trimean", decimal=2)
    np.testing.assert_almost_equal(sine_fts["trimean"], 0.0, err_msg="sine trimean", decimal=2)

    # stat_hist0
    np.testing.assert_almost_equal(const_0_fts["stat_hist0"], 0.0, err_msg="const0 stat_hist0")
    np.testing.assert_almost_equal(const_1_fts["stat_hist0"], 0.0, err_msg="const1 stat_hist0")
    np.testing.assert_almost_equal(const_neg_fts["stat_hist0"], 0.0, err_msg="const neg stat_hist0")
    np.testing.assert_almost_equal(lin_fts["stat_hist0"], 0.2, err_msg="lin stat_hist0", decimal=2)
    np.testing.assert_almost_equal(sine_fts["stat_hist0"], 0.25, err_msg="sine stat_hist0", decimal=2)

    # stat_hist1
    np.testing.assert_almost_equal(const_0_fts["stat_hist1"], 0.0, err_msg="const0 stat_hist1")
    np.testing.assert_almost_equal(const_1_fts["stat_hist1"], 0.0, err_msg="const1 stat_hist1")
    np.testing.assert_almost_equal(const_neg_fts["stat_hist1"], 0.0, err_msg="const neg stat_hist1")
    np.testing.assert_almost_equal(lin_fts["stat_hist1"], 0.2, err_msg="lin stat_hist1", decimal=2)
    np.testing.assert_almost_equal(sine_fts["stat_hist1"], 0.2, err_msg="sine stat_hist1", decimal=2)

    # stat_hist2
    np.testing.assert_almost_equal(const_0_fts["stat_hist2"], 1.0, err_msg="const0 stat_hist2")
    np.testing.assert_almost_equal(const_1_fts["stat_hist2"], 1.0, err_msg="const1 stat_hist2")
    np.testing.assert_almost_equal(const_neg_fts["stat_hist2"], 1.0, err_msg="const neg stat_hist2")
    np.testing.assert_almost_equal(lin_fts["stat_hist2"], 0.2, err_msg="lin stat_hist2", decimal=2)
    np.testing.assert_almost_equal(sine_fts["stat_hist2"], 0.1, err_msg="sine stat_hist2", decimal=2)

    # stat_hist3
    np.testing.assert_almost_equal(const_0_fts["stat_hist3"], 0.0, err_msg="const0 stat_hist3")
    np.testing.assert_almost_equal(const_1_fts["stat_hist3"], 0.0, err_msg="const1 stat_hist3")
    np.testing.assert_almost_equal(const_neg_fts["stat_hist3"], 0.0, err_msg="const neg stat_hist3")
    np.testing.assert_almost_equal(lin_fts["stat_hist3"], 0.2, err_msg="lin stat_hist3", decimal=2)
    np.testing.assert_almost_equal(sine_fts["stat_hist3"], 0.2, err_msg="sine stat_hist3", decimal=2)

    # stat_hist4
    np.testing.assert_almost_equal(const_0_fts["stat_hist4"], 0.0, err_msg="const0 stat_hist4")
    np.testing.assert_almost_equal(const_1_fts["stat_hist4"], 0.0, err_msg="const1 stat_hist4")
    np.testing.assert_almost_equal(const_neg_fts["stat_hist4"], 0.0, err_msg="const neg stat_hist4")
    np.testing.assert_almost_equal(lin_fts["stat_hist4"], 0.2, err_msg="lin stat_hist4", decimal=2)
    np.testing.assert_almost_equal(sine_fts["stat_hist4"], 0.25, err_msg="sine stat_hist4", decimal=2)

    # entropy
    np.testing.assert_almost_equal(const_0_fts["entropy"],  0.0, err_msg="const0 entropy")
    np.testing.assert_almost_equal(const_1_fts["entropy"], 4.60, err_msg="const1 entropy", decimal=2)
    np.testing.assert_almost_equal(const_neg_fts["entropy"], 4.605, err_msg="const neg entropy", decimal=2)
    np.testing.assert_almost_equal(lin_fts["entropy"], 4.406, err_msg="lin entropy", decimal=2)
    np.testing.assert_almost_equal(sine_fts["entropy"], -1.7976931348623157e+308, err_msg="sine entropy", decimal=2)

    print("End check")
