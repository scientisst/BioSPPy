# -*- coding: utf-8 -*-
"""
biosppy.tests.cepstral
-------------------
This module provides methods to test the cepstral feature extraction module.
:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# 3rd party
import numpy as np

# local
from ..features.cepstral import mfcc


def getData(size=100, sampling_rate=100):
    const_0 = np.zeros(size)
    const_1 = np.ones(size)
    const_neg = -1 * np.ones(size)

    f = 5
    x = np.arange(0, size/sampling_rate, 1/sampling_rate)
    sine = np.sin(2 * np.pi * f * x)
    np.random.seed(0)
    sine = sine + np.random.normal(0, 0.5, size)
    lin = np.arange(size)
    sine = +2*np.sin(2 * np.pi * 10 * x)
    return const_0, const_1, const_neg, lin, sine


def test(size=2000, sampling_rate=24410):
    const_0, const_1, const_neg, lin, sine = getData(size, sampling_rate)
    
    const_0_fts = mfcc(const_0, sampling_rate, sampling_rate)["mfcc"]
    const_1_fts = mfcc(const_1, sampling_rate, sampling_rate)["mfcc"]
    const_neg_fts = mfcc(const_neg, sampling_rate, sampling_rate)["mfcc"]
    lin_fts = mfcc(lin, sampling_rate, sampling_rate)["mfcc"]
    sine_fts = mfcc(sine, sampling_rate, sampling_rate)["mfcc"]
    
    np.testing.assert_almost_equal(const0_fts, [-1.00000000e-08, -2.56546322e-08, -4.09905813e-08, -5.56956514e-08, -6.94704899e-08, -8.20346807e-08, -9.31324532e-08, -1.02537889e-07, -1.10059519e-07], err_msg="const0 mfcc", decimal=2)
    np.testing.assert_almost_equal(const1_fts, [ 248.30695301,   88.68699763,   72.82806925,  -63.63692506, -173.93188851, -353.00476906, -535.48391595, -758.99703192, -986.48117664], err_msg="const1 mfcc", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts, [ 248.30695301 ,  88.68699763 ,  72.82806925,  -63.63692506, -173.93188851, -353.00476906 ,-535.48391595, -758.99703192 ,-986.48117664], err_msg="const neg mfcc", decimal=2)
    np.testing.assert_almost_equal(lin_fts, [ 245.32180296  , 81.85925318 ,  64.09605229,  -71.46085216, -177.42474564, -348.46821293 ,-519.31140372, -728.02836367, -938.33424622], err_msg="lin mfcc", decimal=2)
    np.testing.assert_almost_equal(sine_fts, [  253.25355662 ,   99.95100524,    87.16461631 ,  -50.87104034, -168.34395888 , -360.63497483  ,-562.20900582,  -809.9139956, -1065.34792531], err_msg="sine mfcc", decimal=2)

    print("End Check")
