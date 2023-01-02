# -*- coding: utf-8 -*-
"""
biosppy.tests.frequency
-------------------
This module provides methods to test the frequency feature extraction module.
:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# 3rd party
import numpy as np
from matplotlib import pylab as plt
from scipy.signal import square

# local
from ..features.frequency import freq_features


def getData(LEN=100, sampling_rate=100, f=5):
    const0 = np.zeros(LEN)
    const1 = np.ones(LEN)
    constNeg = -1 * np.ones(LEN)

    x = np.arange(0, LEN/sampling_rate, 1/sampling_rate)
    sine = np.sin(2 * np.pi * f * x)
    np.random.seed(0)
    sineWNoise = sine + np.random.normal(0, 0.5, LEN)
    lin = np.arange(LEN)
    sq = square(2 * np.pi * f * x)

    return const0, const1, constNeg, lin, sine, sineWNoise, sq


def test(LEN=60*1, sampling_rate=100, f=5):
    const0, const1, constNeg, lin, sine, sineWNoise, sq = getData(LEN, sampling_rate, f)
    
    sq_fts = freq_features(sq, sampling_rate)
    sine_fts = freq_features(sine, sampling_rate)
    const0_fts = freq_features(const0, sampling_rate)
    const1_fts = freq_features(const1, sampling_rate)
    constNeg_fts = freq_features(constNeg, sampling_rate)
    lin_fts = freq_features(lin, sampling_rate)

    # fundamental_frequency
    np.testing.assert_almost_equal(const0_fts["fundamental_frequency"], 0.0, err_msg="const0 fundamental_frequency")
    np.testing.assert_almost_equal(const1_fts["fundamental_frequency"], 0.0, err_msg="const1 fundamental_frequency")
    np.testing.assert_almost_equal(constNeg_fts["fundamental_frequency"], 0.0, err_msg="const neg fundamental_frequency")
    np.testing.assert_almost_equal(lin_fts["fundamental_frequency"], 0.0, err_msg="lin fundamental_frequency")
    np.testing.assert_almost_equal(sine_fts["fundamental_frequency"], 5.0, err_msg="sine fundamental_frequency")
    np.testing.assert_almost_equal(sq_fts["fundamental_frequency"], 5.0, err_msg="sine fundamental_frequency")

    # sum_harmonics
    np.testing.assert_almost_equal(const0_fts["sum_harmonics"], 0.0, err_msg="const0 sum_harmonics")
    np.testing.assert_almost_equal(const1_fts["sum_harmonics"], 0.0, err_msg="const1 sum_harmonics")
    np.testing.assert_almost_equal(constNeg_fts["sum_harmonics"], 0.0, err_msg="const neg sum_harmonics")
    np.testing.assert_almost_equal(lin_fts["sum_harmonics"], 0.0, err_msg="lin sum_harmonics")
    np.testing.assert_almost_equal(sine_fts["sum_harmonics"], 0.000734, err_msg="sine sum_harmonics", decimal=2)
    np.testing.assert_almost_equal(sq_fts["sum_harmonics"], 0.37001, err_msg="sine sum_harmonics", decimal=2)

    # spectral_roll_on
    np.testing.assert_almost_equal(const1_fts["spectral_roll_on"], 0.0, err_msg="const1 spectral_roll_on")
    np.testing.assert_almost_equal(constNeg_fts["spectral_roll_on"], 0.0, err_msg="const neg spectral_roll_on")
    np.testing.assert_almost_equal(lin_fts["spectral_roll_on"], 0.0, err_msg="lin spectral_roll_on")
    np.testing.assert_almost_equal(sine_fts["spectral_roll_on"], 4.0, err_msg="sine spectral_roll_on")
    np.testing.assert_almost_equal(sq_fts["spectral_roll_on"], 4.0, err_msg="sine spectral_roll_on")

    # spectral_roll_off
    np.testing.assert_almost_equal(const1_fts["spectral_roll_off"], 2.0, err_msg="const1 spectral_roll_off")
    np.testing.assert_almost_equal(constNeg_fts["spectral_roll_off"], 2.0, err_msg="const neg spectral_roll_off")
    np.testing.assert_almost_equal(lin_fts["spectral_roll_off"], 2.0, err_msg="lin spectral_roll_off")
    np.testing.assert_almost_equal(sine_fts["spectral_roll_off"], 6.0, err_msg="sine spectral_roll_off")
    np.testing.assert_almost_equal(sq_fts["spectral_roll_off"], 26.0, err_msg="sine spectral_roll_off")

    print("End Check")
