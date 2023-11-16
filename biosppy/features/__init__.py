# -*- coding: utf-8 -*-
"""
biosppy.features
----------------

This package provides methods to extract common
signal features in:
    * Time domain
    * Frequency domain
    * Time-frequency domain
    * Phase-space domain
    * Cepstral domain

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# compat
from __future__ import absolute_import, division, print_function
# allow lazy loading
from . import frequency, time, time_freq, cepstral, phase_space

