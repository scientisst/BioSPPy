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


def getData(size=100, sampling_rate=100, f=5):
    const_0 = np.zeros(size)
    const_1 = np.ones(size)
    const_neg = -1 * np.ones(size)

    x = np.arange(0, size/sampling_rate, 1/sampling_rate)
    sine = np.sin(2 * np.pi * f * x)
    np.random.seed(0)
    lin = np.arange(size)
    sq = square(2 * np.pi * f * x)

    return const_0, const_1, const_neg, lin, sine, sq


def test(size=60*1, sampling_rate=100, f=5):
    const_0, const_1, const_neg, lin, sine, sq = getData(size, sampling_rate, f)
     
    sq_fts = freq_features(sq, sampling_rate)
    sine_fts = freq_features(sine, sampling_rate)
    const_0_fts = freq_features(const_0, sampling_rate)
    const_1_fts = freq_features(const_1, sampling_rate)
    const_neg_fts = freq_features(const_neg, sampling_rate)
    lin_fts = freq_features(lin, sampling_rate)

    # fundamental_frequency
    np.testing.assert_almost_equal(const_0_fts["fundamental_frequency"], 0.0, err_msg="const0 fundamental_frequency")
    np.testing.assert_almost_equal(const_1_fts["fundamental_frequency"], 0.0, err_msg="const1 fundamental_frequency")
    np.testing.assert_almost_equal(const_neg_fts["fundamental_frequency"], 0.0, err_msg="const neg fundamental_frequency")
    np.testing.assert_almost_equal(lin_fts["fundamental_frequency"], 0.0, err_msg="lin fundamental_frequency")
    np.testing.assert_almost_equal(sine_fts["fundamental_frequency"], 51.7241, err_msg="sine fundamental_frequency", decimal=2)
    np.testing.assert_almost_equal(sq_fts["fundamental_frequency"], 51.7241, err_msg="sine fundamental_frequency", decimal=2)

    # sum_harmonics
    np.testing.assert_array_equal(const_0_fts["sum_harmonics"], None, err_msg="const0 sum_harmonics")
    np.testing.assert_array_equal(const_1_fts["sum_harmonics"], None, err_msg="const1 sum_harmonics")
    np.testing.assert_array_equal(const_neg_fts["sum_harmonics"], None, err_msg="const neg sum_harmonics")
    np.testing.assert_array_equal(lin_fts["sum_harmonics"], None, err_msg="lin sum_harmonics")
    np.testing.assert_array_equal(sine_fts["sum_harmonics"], None, err_msg="sine sum_harmonics")
    np.testing.assert_array_equal(sq_fts["sum_harmonics"], None, err_msg="sine sum_harmonics")

    # spectral_roll_on
    np.testing.assert_almost_equal(const_1_fts["spectral_roll_on"], 0.0, err_msg="const1 spectral_roll_on")
    np.testing.assert_almost_equal(const_neg_fts["spectral_roll_on"], 0.0, err_msg="const neg spectral_roll_on")
    np.testing.assert_almost_equal(lin_fts["spectral_roll_on"], 0.0, err_msg="lin spectral_roll_on")
    np.testing.assert_almost_equal(sine_fts["spectral_roll_on"], 51.72413, err_msg="sine spectral_roll_on", decimal=2)
    np.testing.assert_almost_equal(sq_fts["spectral_roll_on"], 51.72413, err_msg="sine spectral_roll_on", decimal=2)

    # spectral_roll_off
    np.testing.assert_array_equal(const_1_fts["spectral_roll_off"], 0.0, err_msg="const1 spectral_roll_off")
    np.testing.assert_almost_equal(const_neg_fts["spectral_roll_off"], 0.0, err_msg="const neg spectral_roll_off")
    np.testing.assert_almost_equal(lin_fts["spectral_roll_off"], 17.241379, err_msg="lin spectral_roll_off", decimal=2)
    np.testing.assert_almost_equal(sine_fts["spectral_roll_off"], 51.7241379, err_msg="sine spectral_roll_off", decimal=2)
    np.testing.assert_almost_equal(sq_fts["spectral_roll_off"], 51.724, err_msg="sine spectral_roll_off", decimal=2)

    print("End Check")
