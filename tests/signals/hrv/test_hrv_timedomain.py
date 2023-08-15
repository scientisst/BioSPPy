from biosppy.signals.hrv import hrv_timedomain
import pandas as pd
import numpy as np
import pytest
import os


def test_compute_time_domain_features_rri_is_none() -> None:
    
    """Test that an error is 
    raised when rri is None"""

    with pytest.raises(ValueError):
        hrv_timedomain(None)

def test_hrv_timedomain_empty():

    ''' What happens when we 
    provide an empty list to
    the hrv_timedomain '''

    with pytest.raises(ValueError):
        hrv_timedomain(rri = [], detrend_rri = False)


def test_compute_time_domain_features_duration_lt_10s(rri_rpeaks) -> None:

    """Test that an error is raised when 
    duration is less than 10 seconds"""

    rri,_ = rri_rpeaks

    with pytest.raises(ValueError):
        hrv_timedomain(rri, duration=9, detrend_rri = False)


def test_hrv_timedomain_duration_10_20(rri_rpeaks):

    ''' Test if only the parameters
    between 10 - 20 duration are calculated'''

    rri,_ = rri_rpeaks
    results = hrv_timedomain(rri, duration = 15)

    required_keys = ['hr', 'hr_min', 'hr_max', 'hr_minmax', 'hr_mean', 'hr_median',
    'rr_min', 'rr_max', 'rr_minmax', 'rr_mean', 'rr_median', 'rmssd']

    assert all(key in results.keys() for key in required_keys)


def test_hrv_timedomain_20_60(rri_rpeaks):

    ''' Test if only the parameters with
    less than 60 duration are calculated '''

    rri,_ = rri_rpeaks
    results = hrv_timedomain(rri, duration = 25)

    required_keys = ['hr', 'hr_min', 'hr_max', 'hr_minmax', 'hr_mean', 'hr_median',
    'rr_min', 'rr_max', 'rr_minmax', 'rr_mean', 'rr_median', 'rmssd', 'nn50', 'pnn50',
    ]

    assert all(key in results.keys() for key in required_keys)


def test_hrv_timedomain_60_90(rri_rpeaks):

    ''' Test if the parameters 
    less than 90 are calculated '''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, duration =65)

    required_keys = ['hr', 'hr_min', 'hr_max', 'hr_minmax', 'hr_mean', 'hr_median',
    'rr_min', 'rr_max', 'rr_minmax', 'rr_mean', 'rr_median', 'rmssd', 'nn50', 'pnn50', 'sdnn']

    assert all(key in results.keys() for key in required_keys)


def test_hrv_timedomain_90(rri_rpeaks):

    ''' Test if the parameters
    required a minimum duration of 
    90 are calculated '''

    rri, _ = rri_rpeaks

    results = hrv_timedomain(rri, duration =95)

    required_keys = ['hr', 'hr_min', 'hr_max', 'hr_minmax', 'hr_mean', 'hr_median',
    'rr_min', 'rr_max', 'rr_minmax', 'rr_mean', 'rr_median', 'rmssd', 'nn50', 'pnn50', 'sdnn',
    'hti', 'tinn']

    assert all(key in results.keys() for key in required_keys)


def test_hrv_timedomain_hr(rri_rpeaks):

    ''' Measuring instantaneous
    heart rate '''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri = rri, detrend_rri = False)
    assert np.array_equal(results['hr'], 'something') 

def test_hr_min(rri_rpeaks) -> None:

    ''' Testing heart 
    rate minimum '''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)
    assert results['hr_min'] == 50

def test_hr_max(rri_rpeaks) -> None:

    ''' Testing heart
    rate maximum '''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)
    assert results['hr_max'] == 75

def test_hr_minmax(rri_rpeaks) -> None:

    ''' Testing hr minimum 
    and maximum difference '''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)  
    assert results['hr_minmax'] == 25
    
def test_hr_mean(rri_rpeaks) -> None:

    ''' Testing heart 
    rate mean '''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)
    assert results['hr_mean'] == 63.33333333333333
    
def test_hr_median(rri_rpeaks) -> None:
    
    ''' Testing Heart 
    Rate Median'''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)
    assert results['hr_median'] == 66.66666666666667

def test_rri_min(rri_rpeaks) -> None:
    
    ''' Testing minimum
    rr interval '''
    
    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)
    assert results['rr_min'] == 800

def test_rr_max(rri_rpeaks) -> None:
    
    ''' Testing maximum
    rr interval '''
    
    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)
    assert results['rr_max'] == 1300
    
def test_rr_minmax(rri_rpeaks) -> None:
    
    ''' Testing rr interval
    minimum and maximum '''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)
    assert results['rr_minmax'] == 500
    
def test_rr_mean(rri_rpeaks) -> None:

    ''' Testing rr 
    interval mean '''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)
    assert results['rr_mean'] == 1050

def test_rr_median(rri_rpeaks) -> None:
    
    ''' Testing rr 
    interval median '''
    
    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)
    assert results['rr_median'] == 1050

def test_rmssd(rri_rpeaks) -> None:
    
    ''' Testing rmssd '''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)
    assert results['rmssd'] == pytest.approx(111.80339887498948, rel=1e-6)

def test_nn50(rri_rpeaks) -> None:

    ''' Testing nn50 '''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)
    assert results['nn50'] == 2

def test_pnn50(rri_rpeaks) -> None:

    ''' Testing pnn50 '''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)
    assert results['pnn50'] == pytest.approx(33.33333333333333, rel=1e-6)

def test_sdnn(rri_rpeaks) -> None:
    
    ''' Testing SDNN '''
    
    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, duration=60, detrend_rri = False)
    assert results['sdnn'] == pytest.approx(188.9822365046136, rel=1e-6)

def test_geometrical(rri_rpeaks) -> None:
    
    ''' Testing HRV 
    Triangular Index '''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, duration=90, detrend_rri = False)
    assert results['hti'] == pytest.approx(14.6, rel=1e-6)

def test_tinn(rri_rpeaks) -> None:

    ''' Testing TINN '''

    rri, _ = rri_rpeaks
    results = hrv_timedomain(rri, detrend_rri = False)
    assert results['tinn'] == pytest.approx(226.5625, rel=1e-6)
