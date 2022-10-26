import numpy as np
from ..features.quefrency import mfcc


def getData(LEN=100, SR=100):
    const0 = np.zeros(LEN)
    const1 = np.ones(LEN)
    constNeg = -1 * np.ones(LEN)

    f = 5
    x = np.arange(0, LEN/SR, 1/SR)
    sine = np.sin(2 * np.pi * f * x)
    np.random.seed(0)
    sine = sine + np.random.normal(0, 0.5, LEN)
    lin = np.arange(LEN)
    sine = +2*np.sin(2 * np.pi * 10 * x)
    return const0, const1, constNeg, lin, sine


def test(LEN=2000, SR=24410):
    const0, const1, constNeg, lin, sine = getData(LEN, SR)
    
    const0_fts = mfcc(const0, SR, SR)["mfcc"]
    const1_fts = mfcc(const1, SR, SR)["mfcc"]
    constNeg_fts = mfcc(constNeg, SR, SR)["mfcc"]
    lin_fts = mfcc(lin, SR, SR)["mfcc"]
    sine_fts = mfcc(sine, SR, SR)["mfcc"]
    
    ## mfcc
    np.testing.assert_almost_equal(const0_fts, [-1.00000000e-08, -2.56546322e-08, -4.09905813e-08, -5.56956514e-08, -6.94704899e-08, -8.20346807e-08, -9.31324532e-08, -1.02537889e-07, -1.10059519e-07], err_msg="const0 mfcc", decimal=2)
    np.testing.assert_almost_equal(const1_fts, [ 248.30695301,   88.68699763,   72.82806925,  -63.63692506, -173.93188851, -353.00476906, -535.48391595, -758.99703192, -986.48117664], err_msg="const1 mfcc", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts, [ 248.30695301 ,  88.68699763 ,  72.82806925,  -63.63692506, -173.93188851, -353.00476906 ,-535.48391595, -758.99703192 ,-986.48117664], err_msg="const neg mfcc", decimal=2)
    np.testing.assert_almost_equal(lin_fts, [ 245.32180296  , 81.85925318 ,  64.09605229,  -71.46085216, -177.42474564, -348.46821293 ,-519.31140372, -728.02836367, -938.33424622], err_msg="lin mfcc", decimal=2)
    np.testing.assert_almost_equal(sine_fts, [  253.25355662 ,   99.95100524,    87.16461631 ,  -50.87104034, -168.34395888 , -360.63497483  ,-562.20900582,  -809.9139956, -1065.34792531], err_msg="sine mfcc", decimal=2)


