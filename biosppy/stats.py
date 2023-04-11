# -*- coding: utf-8 -*-
"""
biosppy.stats
-------------

This module provides statistical functions and related tools.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
import six

# local
from . import utils
from .signals import tools

# 3rd party
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_rel, ttest_ind


def pearson_correlation(x=None, y=None):
    """Compute the Pearson Correlation Coefficient between two signals.

    The coefficient is given by:

    .. math::

        r_{xy} = \\frac{E[(X - \\mu_X) (Y - \\mu_Y)]}{\\sigma_X \\sigma_Y}

    Parameters
    ----------
    x : array
        First input signal.
    y : array
        Second input signal.

    Returns
    -------
    r : float
        Pearson correlation coefficient, ranging between -1 and +1.
    pvalue : float
        Two-tailed p-value. The p-value roughly indicates the probability of
        an uncorrelated system producing datasets that have a Pearson correlation
        at least as extreme as the one computed from these datasets.

    Raises
    ------
    ValueError
        If the input signals do not have the same length.

    """

    # check inputs
    if x is None:
        raise TypeError("Please specify the first input signal.")

    if y is None:
        raise TypeError("Please specify the second input signal.")

    # ensure numpy
    x = np.array(x)
    y = np.array(y)

    n = len(x)

    if n != len(y):
        raise ValueError("Input signals must have the same length.")

    r, pvalue = pearsonr(x, y)

    args = (r, pvalue)
    names = ('r', 'pvalue')

    return utils.ReturnTuple(args, names)


def linear_regression(x=None, y=None, show=True):
    """Plot the linear regression between two signals and get the equation coefficients.

    The linear regression uses the least squares method.

    Parameters
    ----------
    x : array
        First input signal.
    y : array
        Second input signal.
    show : bool
        If True, show the plot.

    Returns
    -------
    coeffs : array
        Linear regression coefficients: [m, b].

    Raises
    ------
    ValueError
        If the input signals do not have the same length.

    """

    # check inputs
    if x is None:
        raise TypeError("Please specify the first input signal.")

    if y is None:
        raise TypeError("Please specify the second input signal.")

    # ensure numpy
    x = np.array(x)
    y = np.array(y)

    n = len(x)

    if n != len(y):
        raise ValueError("Input signals must have the same length.")

    coeffs = np.polyfit(x, y, 1)
    f = np.poly1d(coeffs)

    x_min = x.min()
    x_max = x.max()

    y_min = f(x_min)
    y_max = f(x_max)

    if show:
        plt.scatter(x, y)
        plt.plot(
            [x_min, x_max],
            [y_min, y_max],
            c="orange",
            label="y={:.3f}x+{:.3f}".format(coeffs[0], coeffs[1]),
        )
        plt.title("Linear Regression")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

    args = (coeffs[0], coeffs[1])
    labels = ["m", "b"]

    return utils.ReturnTuple(args, labels)


def paired_test(x=None, y=None):
    """
    Perform the Student's paired t-test on the arrays x and y.
    This is a two-sided test for the null hypothesis that 2 related
    or repeated samples have identical average (expected) values.

    Parameters
    ----------
    x : array
        First input signal.
    y : array
        Second input signal.

    Returns
    -------
    statistic : float
        t-statistic. The t-statistic is used in a t-test to determine
        if you should support or reject the null hypothesis.
    pvalue : float
        Two-sided p-value.

    Raises
    ------
    ValueError
        If the input signals do not have the same length.

    """

    # check inputs
    if x is None:
        raise TypeError("Please specify the first input signal.")

    if y is None:
        raise TypeError("Please specify the second input signal.")

    # ensure numpy
    x = np.array(x)
    y = np.array(y)

    n = len(x)

    if n != len(y):
        raise ValueError("Input signals must have the same length.")

    statistic, pvalue = ttest_rel(x, y)

    return statistic, pvalue


def unpaired_test(x=None, y=None):
    """
    Perform the Student's unpaired t-test on the arrays x and y.
    This is a two-sided test for the null hypothesis that 2 independent
    samples have identical average (expected) values. This test assumes
    that the populations have identical variances by default.

    Parameters
    ----------
    x : array
        First input signal.
    y : array
        Second input signal.

    Returns
    -------
    statistic : float
        t-statistic. The t-statistic is used in a t-test to determine
        if you should support or reject the null hypothesis.
    pvalue : float
        Two-sided p-value.

    Raises
    ------
    ValueError
        If the input signals do not have the same length.

    """

    # check inputs
    if x is None:
        raise TypeError("Please specify the first input signal.")

    if y is None:
        raise TypeError("Please specify the second input signal.")

    # ensure numpy
    x = np.array(x)
    y = np.array(y)

    n = len(x)

    if n != len(y):
        raise ValueError("Input signals must have the same length.")

    statistic, pvalue = ttest_ind(x, y)

    return statistic, pvalue


def histogram(signal=None, bins=5, normalize=True):
    """Compute histogram of the input signal.

    Parameters
    ----------
    signal : array
        Input signal.
    bins : int, optional
        Number of histogram bins. Default is 5.
    normalize : bool, optional
        Whether to normalize the histogram counts. Default is True.

    Returns
    -------
    hist{bin}_bins : float
        Number of counts of the bin. If `normalize` is True, the counts are normalized.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure input formats
    signal = np.array(signal)
    bins = int(bins)

    # initialize output
    out = utils.ReturnTuple((), ())

    # compute histogram
    hist = np.histogram(signal, bins=bins)[0]
    if normalize:
        hist = hist / np.sum(hist)  # normalization

    # add counts
    for index, count in enumerate(hist):
        out = out.append(count, 'hist_' + str(index+1) + '_' + str(bins))

    return out


def quartiles(signal=None):
    """Compute quartile features of the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    q1 : float
        First quartile.
    q2 : float
        Second quartile, also known as median.
    q3 : float
        Third quartile.
    iqr : float
        Interquartile range.
    midhinge : float
        Midhinge.
    trimean : float
        Trimean.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure input formats
    signal = np.array(signal)

    # initialize output
    out = utils.ReturnTuple((), ())

    # compute quartiles
    q1, q2, q3 = np.quantile(signal, [0.25, 0.5, 0.75])
    out = out.append([q1, q2, q3], ['q1', 'q2', 'q3'])

    # iqr
    iqr = q3 - q1
    out = out.append(iqr, 'iqr')

    # midhinge
    midhinge = (q3 + q1) / 2
    out = out.append(midhinge, 'midhinge')

    # trimean
    trimean = (q2 + midhinge) / 2
    out = out.append(trimean, 'trimean')

    return out


def diff_stats(signal=None, stats_only=True):
    """Compute statistical features from the first signal differences, second
    signal differences and absolute signal differences.

    Parameters
    ----------
    signal : array
        Input signal.
    stats_only : bool, optional
        Whether to output only statistical features. Default is True.

    Returns
    -------
    {diff} : array
        Difference signal. {diff} can be 'diff', 'diff2' or 'abs_diff'.
    {diff}_mean : float
        Mean of the difference signal.
    {diff}_median : float
        Median of the difference signal.
    {diff}_min : float
        Minimum of the difference signal.
    {diff}_max : float
        Maximum of the difference signal.
    {diff}_max_amp : float
        Maximum amplitude of the difference signal.
    {diff}_range : float
        Range of the difference signal.
    {diff}_var : float
        Variance of the difference signal.
    {diff}_std : float
        Standard deviation of the difference signal.
    {diff}_sum : float
        Sum of the difference signal.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # initialize output
    out = utils.ReturnTuple((), ())

    # compute differences
    sig_diff = np.diff(signal)
    sig_diff_2 = np.diff(sig_diff)
    sig_diff_abs = np.abs(sig_diff)

    diffs = [sig_diff, sig_diff_2, sig_diff_abs]
    labels = ['firstdiff', 'seconddiff', 'absdiff']

    # extract features
    for diff, label in zip(diffs, labels):
        # add to output
        if not stats_only:
            out = out.append(diff, label)

        # compute stats
        diff_stat = tools.signal_stats(diff)

        # add to output
        for arg, name in zip(diff_stat, diff_stat.keys()):
            out = out.append(arg, label + '_' + name)

        # sum
        sum_ = np.sum(diff)
        out = out.append(sum_, label + '_' + 'sum')

    return out
