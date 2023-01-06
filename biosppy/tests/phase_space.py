# -*- coding: utf-8 -*-
"""
biosppy.tests.phase_space
-------------------
This module provides methods to test the phase space feature extraction module.
:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# 3rd party
import numpy as np
from matplotlib import pylab as plt

# local
from ..features.phase_space import phase_space_features


def getData(size=100, sampling_rate=100):
    const0 = np.zeros(size)
    const1 = np.ones(size)
    constNeg = -1 * np.ones(size)

    f = 5
    x = np.arange(0, size/sampling_rate, 1/sampling_rate)
    sine = np.sin(2 * np.pi * f * x)
    np.random.seed(0)
    sineWNoise = sine + np.random.normal(0, 0.5, size)
    lin = np.arange(size)
    
    return const0, const1, constNeg, lin, sine


def test(size=100, sampling_rate=100):
    const0, const1, constNeg, lin, sine = getData(size, sampling_rate)
    
    const0_fts = phase_space_features(const0)
    const1_fts = phase_space_features(const1)
    constNeg_fts = phase_space_features(constNeg)
    lin_fts = phase_space_features(lin)
    sine_fts = phase_space_features(sine)

    # rec_plt_rec_rate
    np.testing.assert_almost_equal(const0_fts["rec_plot_rec_rate"],  1.0, err_msg="const0 rec_plt_rec_rate")
    np.testing.assert_almost_equal(const1_fts["rec_plot_rec_rate"], 1.00, err_msg="const1 rec_plt_rec_rate", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts["rec_plot_rec_rate"], 1.0, err_msg="const neg rec_plt_rec_rate", decimal=2)
    np.testing.assert_almost_equal(lin_fts["rec_plot_rec_rate"], 0.01, err_msg="lin rec_plt_rec_rate", decimal=2)
    np.testing.assert_almost_equal(sine_fts["rec_plot_rec_rate"], 0.38, err_msg="sine rec_plt_rec_rate", decimal=2)

    # rec_plt_determ
    np.testing.assert_almost_equal(const0_fts["rec_plot_determ"],  1.0, err_msg="const0 rec_plt_determ")
    np.testing.assert_almost_equal(const1_fts["rec_plot_determ"], 1.0, err_msg="const1 rec_plt_determ", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts["rec_plot_determ"], 1.0, err_msg="const neg rec_plt_determ", decimal=2)
    np.testing.assert_almost_equal(lin_fts["rec_plot_determ"], 0.65, err_msg="lin rec_plt_determ", decimal=2)
    np.testing.assert_almost_equal(sine_fts["rec_plot_determ"], 0.99, err_msg="sine rec_plt_determ", decimal=2)

    # rec_plot_laminarity
    np.testing.assert_almost_equal(const0_fts["rec_plot_laminarity"],  1.0, err_msg="const0 rec_plt_laminarity")
    np.testing.assert_almost_equal(const1_fts["rec_plot_laminarity"], 1.0, err_msg="const1 rec_plt_laminarity", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts["rec_plot_laminarity"], 1.0, err_msg="const neg rec_plt_laminarity", decimal=2)
    np.testing.assert_almost_equal(lin_fts["rec_plot_laminarity"], 0.71, err_msg="lin rec_plt_laminarity", decimal=2)
    np.testing.assert_almost_equal(sine_fts["rec_plot_laminarity"], 0.99, err_msg="sine rec_plt_laminarity", decimal=2)

    # rec_plot_det_rr_ratio
    np.testing.assert_almost_equal(const0_fts["rec_plot_det_rr_ratio"],  1.0, err_msg="const0 rec_plot_det_rr_ratio", decimal=2)
    np.testing.assert_almost_equal(const1_fts["rec_plot_det_rr_ratio"], 1.0, err_msg="const1 rec_plot_det_rr_ratio", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts["rec_plot_det_rr_ratio"], 1.0, err_msg="const neg rec_plot_det_rr_ratio", decimal=2)
    np.testing.assert_almost_equal(lin_fts["rec_plot_det_rr_ratio"], 86.65, err_msg="lin rec_plot_det_rr_ratio", decimal=2)
    np.testing.assert_almost_equal(sine_fts["rec_plot_det_rr_ratio"], 2.63, err_msg="sine rec_plot_det_rr_ratio", decimal=2)

    # rec_plot_lami_determ_ratio
    np.testing.assert_almost_equal(const0_fts["rec_plot_lami_determ_ratio"],  1.0, err_msg="const0 rec_plot_lami_determ_ratio", decimal=2)
    np.testing.assert_almost_equal(const1_fts["rec_plot_lami_determ_ratio"], 1.0, err_msg="const1 rec_plot_lami_determ_ratio", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts["rec_plot_lami_determ_ratio"], 1.0, err_msg="const neg rec_plot_lami_determ_ratio", decimal=2)
    np.testing.assert_almost_equal(lin_fts["rec_plot_lami_determ_ratio"], 1.09, err_msg="lin rec_plot_lami_determ_ratio", decimal=2)
    np.testing.assert_almost_equal(sine_fts["rec_plot_lami_determ_ratio"], 1.00, err_msg="sine rec_plot_lami_determ_ratio", decimal=2)
    ## rec_plot_avg_diag_line_len
    np.testing.assert_almost_equal(const0_fts["rec_plot_avg_diag_line_len"],  112.5, err_msg="const0 rec_plot_avg_diag_line_len", decimal=2)
    np.testing.assert_almost_equal(const1_fts["rec_plot_avg_diag_line_len"], 112.5, err_msg="const1 rec_plot_avg_diag_line_len", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts["rec_plot_avg_diag_line_len"], 112.5, err_msg="const neg rec_plot_avg_diag_line_len", decimal=2)
    np.testing.assert_almost_equal(lin_fts["rec_plot_avg_diag_line_len"], 0.005, err_msg="lin rec_plot_avg_diag_line_len", decimal=2)
    np.testing.assert_almost_equal(sine_fts["rec_plot_avg_diag_line_len"], 0.577, err_msg="sine rec_plot_avg_diag_line_len", decimal=2)

    # rec_plot_avg_vert_line_len
    np.testing.assert_almost_equal(const0_fts["rec_plot_avg_vert_line_len"],  224.0, err_msg="const0 rec_plot_avg_vert_line_len", decimal=2)
    np.testing.assert_almost_equal(const1_fts["rec_plot_avg_vert_line_len"], 224.0, err_msg="const1 rec_plot_avg_vert_line_len", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts["rec_plot_avg_vert_line_len"], 224.0, err_msg="const neg rec_plot_avg_vert_line_len", decimal=2)
    np.testing.assert_almost_equal(lin_fts["rec_plot_avg_vert_line_len"], 0.008, err_msg="lin rec_plot_avg_vert_line_len", decimal=2)
    np.testing.assert_almost_equal(sine_fts["rec_plot_avg_vert_line_len"], 0.592, err_msg="sine rec_plot_avg_vert_line_len", decimal=2)
  
    # rec_plot_avg_white_vert_line_len
    np.testing.assert_almost_equal(const0_fts["rec_plot_avg_white_vert_line_len"],  0.0, err_msg="const0 rec_plot_avg_white_vert_line_len", decimal=2)
    np.testing.assert_almost_equal(const1_fts["rec_plot_avg_white_vert_line_len"], 0.0, err_msg="const1 rec_plot_avg_white_vert_line_len", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts["rec_plot_avg_white_vert_line_len"], 0.0, err_msg="const neg rec_plot_avg_white_vert_line_len", decimal=2)
    np.testing.assert_almost_equal(lin_fts["rec_plot_avg_white_vert_line_len"], 42.88, err_msg="lin rec_plot_avg_white_vert_line_len", decimal=2)
    np.testing.assert_almost_equal(sine_fts["rec_plot_avg_white_vert_line_len"], 1.478, err_msg="sine rec_plot_avg_white_vert_line_len", decimal=2)

    # rec_plot_trapping_tm
    np.testing.assert_almost_equal(const0_fts["rec_plot_trapping_tm"],  224.0, err_msg="const0 rec_plot_trapping_tm", decimal=2)
    np.testing.assert_almost_equal(const1_fts["rec_plot_trapping_tm"], 224.0, err_msg="const1 rec_plot_trapping_tm", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts["rec_plot_trapping_tm"], 224.0, err_msg="const neg rec_plot_trapping_tm", decimal=2)
    np.testing.assert_almost_equal(lin_fts["rec_plot_trapping_tm"], 0.00, err_msg="lin rec_plot_trapping_tm", decimal=2)
    np.testing.assert_almost_equal(sine_fts["rec_plot_trapping_tm"], 0.592, err_msg="sine rec_plot_trapping_tm", decimal=2)

    # rec_plot_lgst_vert_line_len
    np.testing.assert_almost_equal(const0_fts["rec_plot_lgst_vert_line_len"], 224.0, err_msg="const0 rec_plot_lgst_vert_line_len", decimal=2)
    np.testing.assert_almost_equal(const1_fts["rec_plot_lgst_vert_line_len"], 224.0, err_msg="const1 rec_plot_lgst_vert_line_len", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts["rec_plot_lgst_vert_line_len"], 224.0, err_msg="const neg rec_plot_lgst_vert_line_len", decimal=2)
    np.testing.assert_almost_equal(lin_fts["rec_plot_lgst_vert_line_len"], 4.0, err_msg="lin rec_plot_lgst_vert_line_len", decimal=2)
    np.testing.assert_almost_equal(sine_fts["rec_plot_lgst_vert_line_len"], 23.0, err_msg="sine rec_plot_lgst_vert_line_len", decimal=2)

    # rec_plot_entropy_vert_line
    np.testing.assert_almost_equal(const0_fts["rec_plot_entropy_vert_line"], 0.0, err_msg="const0 rec_plot_entropy_vert_line", decimal=2)
    np.testing.assert_almost_equal(const1_fts["rec_plot_entropy_vert_line"], 0.0, err_msg="const1 rec_plot_entropy_vert_line", decimal=2)
    np.testing.assert_almost_equal(constNeg_fts["rec_plot_entropy_vert_line"], 0.0, err_msg="const neg rec_plot_entropy_vert_line", decimal=2)
    np.testing.assert_almost_equal(lin_fts["rec_plot_entropy_vert_line"], 0.0, err_msg="lin rec_plot_entropy_vert_line", decimal=2)
    np.testing.assert_almost_equal(sine_fts["rec_plot_entropy_vert_line"], 0.0, err_msg="sine rec_plot_entropy_vert_line", decimal=2)

    print("End Check")
