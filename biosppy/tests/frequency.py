import numpy as np
from matplotlib import pylab as plt
from ..features.frequency import freq_features
from scipy.signal import square

def getData(LEN=100, SR=100, f=5):
    const0 = np.zeros(LEN)
    const1 = np.ones(LEN)
    constNeg = -1 * np.ones(LEN)

    x = np.arange(0, LEN/SR, 1/SR)
    sine = np.sin(2 * np.pi * f * x)
    np.random.seed(0)
    sineWNoise = sine + np.random.normal(0, 0.5, LEN)
    lin = np.arange(LEN)
    sq = square(2 * np.pi * f * x)
    #plt.figure()
    #plt.plot(const0, label="const0")
    #plt.plot(const1, label="const1")
    #plt.plot(constNeg, label="constNeg")
    #plt.plot(lin, label="lin")
    #plt.plot(sine, label="sine")
    #plt.legend()
    #plt.show()

    return const0, const1, constNeg, lin, sine, sineWNoise, sq


def test(LEN=60*1, SR=100, f=5):
    const0, const1, constNeg, lin, sine, sineWNoise, sq = getData(LEN, SR, f)
    
    sq_fts = freq_features(sq, SR)
    sine_fts = freq_features(sine, SR)
    const0_fts = freq_features(const0, SR)
    const1_fts = freq_features(const1, SR)
    constNeg_fts = freq_features(constNeg, SR)
    lin_fts = freq_features(lin, SR)

    ## fundamental_frequency
    np.testing.assert_almost_equal(const0_fts["fundamental_frequency"], 0.0, err_msg="const0 fundamental_frequency")
    np.testing.assert_almost_equal(const1_fts["fundamental_frequency"], 0.0, err_msg="const1 fundamental_frequency")
    np.testing.assert_almost_equal(constNeg_fts["fundamental_frequency"], 0.0, err_msg="const neg fundamental_frequency")
    np.testing.assert_almost_equal(lin_fts["fundamental_frequency"], 0.0, err_msg="lin fundamental_frequency")
    np.testing.assert_almost_equal(sine_fts["fundamental_frequency"], 5.0, err_msg="sine fundamental_frequency")
    np.testing.assert_almost_equal(sq_fts["fundamental_frequency"], 5.0, err_msg="sine fundamental_frequency")

    ## sum_harmonics
    np.testing.assert_almost_equal(const0_fts["sum_harmonics"], 0.0, err_msg="const0 sum_harmonics")
    np.testing.assert_almost_equal(const1_fts["sum_harmonics"], 0.0, err_msg="const1 sum_harmonics")
    np.testing.assert_almost_equal(constNeg_fts["sum_harmonics"], 0.0, err_msg="const neg sum_harmonics")
    np.testing.assert_almost_equal(lin_fts["sum_harmonics"], 0.0, err_msg="lin sum_harmonics")
    np.testing.assert_almost_equal(sine_fts["sum_harmonics"], 0.000734, err_msg="sine sum_harmonics", decimal=2)
    np.testing.assert_almost_equal(sq_fts["sum_harmonics"], 0.37001, err_msg="sine sum_harmonics", decimal=2)

    ## spectral_roll_on
    #np.testing.assert_almost_equal(const0_fts["spectral_roll_on"], None, err_msg="const0 spectral_roll_on")
    np.testing.assert_almost_equal(const1_fts["spectral_roll_on"], 0.0, err_msg="const1 spectral_roll_on")
    np.testing.assert_almost_equal(constNeg_fts["spectral_roll_on"], 0.0, err_msg="const neg spectral_roll_on")
    np.testing.assert_almost_equal(lin_fts["spectral_roll_on"], 0.0, err_msg="lin spectral_roll_on")
    np.testing.assert_almost_equal(sine_fts["spectral_roll_on"], 4.0, err_msg="sine spectral_roll_on")
    np.testing.assert_almost_equal(sq_fts["spectral_roll_on"], 4.0, err_msg="sine spectral_roll_on")

    ## spectral_roll_off
    #np.testing.assert_almost_equal(const0_fts["spectral_roll_off"], None, err_msg="const0 spectral_roll_off")
    np.testing.assert_almost_equal(const1_fts["spectral_roll_off"], 2.0, err_msg="const1 spectral_roll_off")
    np.testing.assert_almost_equal(constNeg_fts["spectral_roll_off"], 2.0, err_msg="const neg spectral_roll_off")
    np.testing.assert_almost_equal(lin_fts["spectral_roll_off"], 2.0, err_msg="lin spectral_roll_off")
    np.testing.assert_almost_equal(sine_fts["spectral_roll_off"], 6.0, err_msg="sine spectral_roll_off")
    np.testing.assert_almost_equal(sq_fts["spectral_roll_off"], 26.0, err_msg="sine spectral_roll_off")
