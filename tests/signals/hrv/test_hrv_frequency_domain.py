from biosppy.signals.hrv import hrv_frequencydomain
from biosppy.signals.hrv import compute_fbands
import pandas as pd
import numpy as np
import pytest
import os


def test_compute_hrv_frequency_none() -> None:

    ''' When rri is None, 
    return TypeError '''

    with pytest.raises(TypeError):
        hrv_frequencydomain()

def test_compute_hrv_frequency_invalid(rri_rpeaks) -> None:

    ''' Raised Error 
    when the sequence 
    is invalid '''
    
    rri, _ = rri_rpeaks
    with pytest.raises(ValueError):
        hrv_frequencydomain(rri, freq_method='invalid_method')

def test_compute_hrv_frequency_empty() -> None:

    ''' Ensuring it gives error
    when provided an empty list '''

    with pytest.raises(ValueError):
        hrv_frequencydomain(rri = [])

def test_compute_hrv_frequency_noduration(dummy) -> None:

    ''' Ensuring it gives error
    when provided a short rri list '''

    with pytest.raises(ValueError):
        hrv_frequencydomain(rri = dummy)

def test_compute_hrv_frequency_lt20(rri_rpeaks) -> None:

    ''' Check if it raises error
    when provided with an rri list
    defining the duratio as 10 '''
    
    rri, _ = rri_rpeaks
    with pytest.raises(ValueError):
        hrv_frequencydomain(rri, duration = 10)

def test_ulf_pwr(rri_rpeaks) -> None:

    ''' Check if ulf_pwr '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['ulf_pwr'] == pytest.approx(0.03, rel=1e-2)

def test_ulf_peak(rri_rpeaks) -> None:

    ''' Check ulf_pwr '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['ulf_peak'] == pytest.approx(2043.0123, rel=1e-2)

def test_ulf_rpwr(rri_rpeaks) -> None:

    ''' Check ulf_rpwr '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['ulf_rpwr'] == pytest.approx(0.2, rel=1e-2)

def test_vlf_pwr(rri_rpeaks) -> None:

    ''' Check vlf_pwr '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['vlf_pwr'] == pytest.approx(0.2, rel=1e-2)

def test_vlf_peak(rri_rpeaks) -> None:

    ''' Check vlf_peak '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['vlf_peak'] == pytest.approx(0.2, rel=1e-2)

def test_vlf_rpwr(rri_rpeaks) -> None:

    ''' Check vlf_rpwr'''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['vlf_rpwr'] == pytest.approx(0.2, rel=1e-2)

def test_lf_pwr(rri_rpeaks) -> None:

    ''' Check lf_pwr '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['lf_pwr'] == pytest.approx(0.2, rel=1e-2)

def test_lf_peak(rri_rpeaks) -> None:

    ''' Check lf_peak '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['lf_peak'] == pytest.approx(0.2, rel=1e-2)

def test_lf_rpwr(rri_rpeaks) -> None:

    ''' Check lf_rpwr '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['lf_rpwr'] == pytest.approx(0.2, rel=1e-2)

def test_hf_pwr(rri_rpeaks) -> None:

    ''' Check hf_pwr '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['hf_pwr'] == pytest.approx(0.2, rel=1e-2)

def test_hf_peak(rri_rpeaks) -> None:

    ''' Check hf_peak '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['hf_peak'] == pytest.approx(0.2, rel=1e-2)

def test_hf_rpwr(rri_rpeaks) -> None:

    ''' Check hf_rpwr '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['hf_rpwr'] == pytest.approx(0.2, rel=1e-2)

def test_vhf_pwr(rri_rpeaks) -> None:

    ''' Check vhf_pwr '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['vhf_pwr'] == pytest.approx(0.2, rel=1e-2)

def test_vhf_peak(rri_rpeaks) -> None:

    ''' Check vhf_peak '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['vhf_peak'] == pytest.approx(0.2, rel=1e-2)

def test_vhf_rpwr(rri_rpeaks) -> None:

    ''' Check vhf_rpwr '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['vhf_rpwr'] == pytest.approx(0.2, rel=1e-2)

def test_lf_hf(rri_rpeaks) -> None:

    ''' Check lf_hf '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['lf_hf'] == pytest.approx(0.2, rel=1e-2)

def test_lf_nu(rri_rpeaks) -> None:

    ''' Check lf_nu '''

    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['lf_nu'] == pytest.approx(0.2, rel=1e-2)

def test_hf_nu(rri_rpeaks) -> None:
    
    ''' Check hf_nu '''
    
    rri, _ = rri_rpeaks
    results = hrv_frequencydomain(rri)
    assert results['hf_nu'] == pytest.approx(0.2, rel=1e-2)


#### Testing compute_fbands ####
#### Perguntar naq reunião ideias de como
# testar esta função