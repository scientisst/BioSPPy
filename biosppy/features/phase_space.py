# -*- coding: utf-8 -*-
"""
biosppy.features.phase_space
----------------------------

This module provides methods to extract phase-space features.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# 3rd party
import numpy as np
from scipy.signal import resample
from scipy.spatial.distance import pdist, squareform

# local
from .. import utils

# variables
MIN = 2


def phase_space(signal=None):
    """Compute phase-space features describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    feats : ReturnTuple object
        Phase-space features of the signal.

    Notes
    -----
    Check biosppy.features.phase_space.recurrence_plot_features for the list of
    available features.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # initialize output
    feats = utils.ReturnTuple((), ())

    # compute recurrence plot
    rp = compute_recurrence_plot(signal)["rec_matrix"]

    # compute recurrence plot features
    rp_feats = recurrence_plot_features(rp)
    feats = feats.join(rp_feats)

    return feats


def compute_recurrence_plot(signal=None, out_dim=224):
    """Compute recurrence plot (distance matrix).

    Parameters
    ----------
    signal : array
        Input signal.
    out_dim : int, optional
        Output dimension of the recurrence plot.
    
    Returns
    -------
    rec_matrix : array
        Recurrence plot matrix.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # resample to out_dim points
    sig_down = resample(signal, out_dim)
    d = pdist(sig_down[:,None])
    rec = squareform(d)

    args = (rec,)
    names = ('rec_matrix',)

    return utils.ReturnTuple(args, names)


def recurrence_plot_features(rec_matrix=None):
    """Compute recurrence plot features.

    The following features are based on the GitHub repository by bmfreis:
    https://github.com/bmfreis/recurrence_python

    Parameters
    ----------
    rec_matrix : array
        Recurrence plot matrix.

    Returns
    -------
    rec_rate : float
        Recurrence rate: the percentage of recurrence points.
    rec_determ : float
        Recurrence determinism: the percentage of recurrence points which form diagonal lines.
    rec_lamin : float
        Recurrence laminarity: the percentage of recurrence points which form vertical lines.
    rec_determ_rec_rate_ratio : float
        Determinism/recurrence rate ratio.
    rec_lamin_determ_ratio : float
        Laminarity/determinism ratio.
    rec_avg_diag_line_len : float
        Average length of the diagonal lines.
    rec_avg_vert_line_len : float
        Average length of the vertical lines.
    rec_avg_white_vert_line_len : float
        Average length of the white vertical lines.
    rec_plot_trapping_tm : float
        Trapping time.
    rec_plot_lgst_diag_line_len : float
        Length of the longest diagonal line.
    rec_plot_lgst_vert_line_len : float
        Length of the longest vertical line.
    rec_plot_lgst_white_vert_line_len : float
        Length of the longest white vertical line.
    rec_plot_entropy_diag_line : float
        Entropy of the probability distribution of the diagonal line lengths.
    rec_plot_entropy_vert_line : float.
         Entropy of the probability distribution of the vert line lengths.
    rec_plot_entropy_white_vert_line : float
        Entropy of the probability distribution of the white vert line lengths.

    """

    # check inputs
    if rec_matrix is None:
        raise TypeError("Please specify a recurrence plot matrix.")

    # ensure numpy
    rec_matrix = np.array(rec_matrix)

    # initialize output
    feats = utils.ReturnTuple((), ())

    # compute features
    threshold = 0.5
    len_rec_matrix = len(rec_matrix)
    for l in range(len_rec_matrix):
        for c in range(len_rec_matrix):
            rec_matrix[l, c] = 1 if rec_matrix[l, c] < threshold else 0
    len_rec_matrix = len(rec_matrix)
    diag_freq = np.zeros(len_rec_matrix + 1, dtype=int)

    # upper diagonal
    for k in range(1, len_rec_matrix - 1, 1):
        d = np.diag(rec_matrix, k=k)
        d_l = 0
        for _, i in enumerate(d):
            if i:  # has a dot
                d_l += 1
                # if its end of line, finishes counting and adds to hist
                if _ == (len(d) - 1):
                    diag_freq[d_l] += 1
            else:  # doesn't have a dot
                if d_l != 0:
                    diag_freq[d_l] += 1
                # if it's not end of the line and d_l != 0, line ended
                d_l = 0
                diag_freq[d_l] += 1

    # lower diagonal
    for k in range(-1, -(len_rec_matrix - 1), -1):
        d = np.diag(rec_matrix, k=k)
        d_l = 0
        for _, i in enumerate(d):
            if i:  # has a dot
                d_l += 1
                # if its end of line, finishes counting and adds to hist
                if _ == (len(d) - 1):
                    diag_freq[d_l] += 1
            else:  # doesn't have a dot
                if d_l != 0:
                    diag_freq[d_l] += 1
                # if it's not end of the line and d_l != 0, line ended
                d_l = 0
                diag_freq[d_l] += 1

    # vertical lines
    vert_freq = np.zeros(len_rec_matrix + 1, dtype=int)
    for k in range(len_rec_matrix):
        d = rec_matrix[:, k]
        d_l = 0
        for _, i in enumerate(d):
            if i:  # has a dot
                d_l += 1
                # if its end of line, finishes counting and adds to hist
                if _ == (len(d) - 1):
                    vert_freq[d_l] += 1
            else:  # doesn't have a dot
                if d_l != 0:
                    vert_freq[d_l] += 1
                # if it's not end of the line and d_l != 0, line ended
                d_l = 0
                vert_freq[d_l] += 1

    # white vertical lines
    white_vert_freq = np.zeros(len_rec_matrix + 1, dtype=int)
    for k in range(len_rec_matrix):
        d = rec_matrix[:, k]
        d_l = 0
        for _, i in enumerate(d):
            if i == 0:  # has a dot
                d_l += 1
                # if its end of line, finishes counting and adds to hist
                if _ == (len(d) - 1):
                    white_vert_freq[d_l] += 1
            else:  # doesn't have a dot
                if d_l != 0:
                    white_vert_freq[d_l] += 1
                # if it's not end of the line and d_l != 0, line ended
                d_l = 0
                white_vert_freq[d_l] += 1

    # extract features
    # recurrence rate
    rec_rate = 0
    for l in range(len_rec_matrix):
        rec_rate += np.sum(rec_matrix[l])
    rec_rate /= (len_rec_matrix ** 2)
    feats = feats.append(rec_rate, 'rec_rate')

    # determinism
    _sum = np.sum([i * diag_freq[i] for i in range(1, len(diag_freq), 1)])
    if _sum > 0:
        determ = np.sum([i * diag_freq[i] for i in range(MIN, len(diag_freq), 1)]) / _sum
    else:
        determ = None
    feats = feats.append(determ, 'rec_determ')

    # laminarity
    _sum = np.sum([i * vert_freq[i] for i in range(len(vert_freq))])
    if _sum > 0:
        lamin = np.sum([i * vert_freq[i] for i in range(MIN, len(vert_freq), 1)]) / _sum
    else:
        lamin = None
    feats = feats.append(lamin, 'rec_lamin')

    # determinism/recurrence rate ratio
    _sum = np.sum([i * diag_freq[i] for i in range(1, len(diag_freq), 1)])
    if _sum > 0:
        det_rr_ratio = len_rec_matrix ** 2 * (np.sum([i * diag_freq[i] for i in range(MIN, len(diag_freq), 1)]) / _sum ** 2)
    else:
        det_rr_ratio = None
    feats = feats.append(det_rr_ratio, 'rec_determ_rec_rate_ratio')

    # laminarity/determinism ratio
    if determ is not None and determ > 0:
        lamin_determ_ratio = lamin / determ
    else:
        lamin_determ_ratio = None
    feats = feats.append(lamin_determ_ratio, 'rec_lamin_determ_ratio')

    # average diagonal line length
    _sum = np.sum(diag_freq)
    if _sum > 0:
        avg_diag_line_len = np.sum([i * diag_freq[i] for i in range(MIN, len(diag_freq), 1)]) / _sum
    else:
        avg_diag_line_len = None
    feats = feats.append(avg_diag_line_len, 'rec_avg_diag_line_len')

    # average vertical line length
    _sum = np.sum(vert_freq)
    if _sum > 0:
        avg_vert_line_len = np.sum([i * vert_freq[i] for i in range(MIN, len(vert_freq), 1)]) / _sum
    else:
        avg_vert_line_len = None
    feats = feats.append(avg_vert_line_len, 'rec_avg_vert_line_len')

    # average white vertical line length
    _sum = np.sum(white_vert_freq)
    if _sum > 0:
        avg_white_vert_line_len = np.sum([i * white_vert_freq[i] for i in range(MIN, len(white_vert_freq), 1)]) / _sum
    else:
        avg_white_vert_line_len = None
    feats = feats.append(avg_white_vert_line_len, 'rec_avg_white_vert_line_len')

    # trapping time
    _sum = np.sum(vert_freq)
    if _sum > 0:
        rec_plot_trapping_tm = np.sum([i * vert_freq[i] for i in range(MIN, len(vert_freq), 1)]) / _sum
    else:
        rec_plot_trapping_tm = None
    feats = feats.append(rec_plot_trapping_tm, 'rec_plot_trapping_tm')

    # longest diagonal line length
    i_ll = np.sign(diag_freq)
    rec_plot_lgst_diag_line_len = np.where(i_ll == 1)[0][-1]
    feats = feats.append(rec_plot_lgst_diag_line_len, 'rec_plot_lgst_diag_line_len')

    # longest vertical line length
    i_ll = np.sign(vert_freq)
    rec_plot_lgst_vert_line_len = np.where(i_ll == 1)[0][-1]
    feats = feats.append(rec_plot_lgst_vert_line_len, 'rec_plot_lgst_vert_line_len')

    # longest white vertical line length
    i_ll = np.sign(white_vert_freq)
    rec_plot_lgst_white_vert_line_len = np.where(i_ll == 1)[0][-1]
    feats = feats.append(rec_plot_lgst_white_vert_line_len, 'rec_plot_lgst_white_vert_line_len')

    # entropy of diagonal lines
    rec_plot_entropy_diag_line = - np.sum(
        [diag_freq[i] * np.log(diag_freq[i]) if diag_freq[i] > 0 else 0 for i in range(MIN, len(diag_freq), 1)])
    feats = feats.append(rec_plot_entropy_diag_line, 'rec_plot_entropy_diag_line')

    # entropy of vertical lines
    rec_plot_entropy_vert_line = - np.sum(
        [vert_freq[i] * np.log(vert_freq[i]) if vert_freq[i] > 0 else 0 for i in range(MIN, len(vert_freq), 1)])
    feats = feats.append(rec_plot_entropy_vert_line, 'rec_plot_entropy_vert_line')

    # entropy of white vertical lines
    rec_plot_entropy_white_vert_line = - np.sum(
        [white_vert_freq[i] * np.log(white_vert_freq[i]) if white_vert_freq[i] > 0 else 0 for i in
         range(MIN, len(white_vert_freq), 1)])
    feats = feats.append(rec_plot_entropy_white_vert_line, 'rec_plot_entropy_white_vert_line')

    return feats
