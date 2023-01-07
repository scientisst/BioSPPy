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

    np.testing.assert_almost_equal(const_0_fts, [-1.00000000e-08, -2.56546322e-08, -4.09905813e-08, -5.56956514e-08, -6.94704899e-08, -8.20346807e-08, -9.31324532e-08, -1.02537889e-07, -1.10059519e-07], err_msg="const0 mfcc", decimal=2)
    np.testing.assert_almost_equal(const_1_fts, [ -4094.7587913,   -4655.24449943,   3104.0668423,  13786.7513218, 18916.80269864,  13850.59412923,   1693.48355436,  -9058.78575545, -11369.75300578], err_msg="const1 mfcc", decimal=2)
    np.testing.assert_almost_equal(const_neg_fts,  [ -4094.7587913,   -4655.24449943 ,  3104.06684236 , 13786.75132184, 18916.80269864  ,13850.59412923  , 1693.48355436,  -9058.78575545 , -11369.75300578], err_msg="const neg mfcc", decimal=2)
    np.testing.assert_almost_equal(lin_fts, [ 1483.88201703 , 1187.9059237 , -1452.84354142, -2837.49585829,
 -1349.86719777  , 416.79597194 , -683.9139044 , -3868.546269, -5379.7824041 ] , err_msg="lin mfcc", decimal=2)
    np.testing.assert_almost_equal(sine_fts,[  175.74366035  , 433.09709603,   510.77151459 ,  231.21002646,
  -317.02382365 , -792.04194275 , -989.60169899, -1138.51514325, -1664.17796913],  err_msg="sine mfcc", decimal=2)

    print("End Check")
