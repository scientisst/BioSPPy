from biosppy.signals.hrv import hrv_nonlinear
from biosppy.signals.hrv import sample_entropy
from biosppy.signals.hrv import approximate_entropy
import pandas as pd
import numpy as np
import pytest



def test_hrv_nonlinear_missing_input() -> None:

    ''' Check for the missing 
    input in hrv_nonlinear '''

    with pytest.raises(TypeError):
        hrv_nonlinear()

def test_hrv_nonlinear_empty() -> None:

    ''' Verify if for when
    we provide an ampty list for rri '''

    with pytest.raises(TypeError):
        hrv_nonlinear(rri = [])

def test_hrv_nonlinear_short_duration(rri_rpeaks) -> None:
   
   ''' Checking if although providing 
   a valid input rri, if duration is less
   if it raises a value error '''
   
   rri, _ = rri_rpeaks
   with pytest.raises(ValueError):
        hrv_nonlinear(rri, duration=1)


def test_hrv_nonlinear_short_input(dummy) -> None:

    ''' Check the error for 
    a short rri '''

    with pytest.raises(ValueError):
        hrv_nonlinear(rri = dummy)


# Verify Values


def test_hrv_nonlinear_s(rri_rpeaks) -> None:

    ''' Check fro the 
    Poincare Area '''

    rri, _ = rri_rpeaks 
    results = hrv_nonlinear(rri)
    assert results['s'] == pytest.approx(0.0002, 1e-4)

def test_hrv_nonlinear_sd1(rri_rpeaks) -> None:

    ''' Check for sd1 '''

    rri, _ = rri_rpeaks
    results = hrv_nonlinear(rri)
    assert results['sd1'] == pytest.approx(0.0002, 1e-4)

def test_hrv_nonlinear_sd2(rri_rpeaks) -> None:
    
    ''' Check for sd2 '''
    
    rri, _ = rri_rpeaks
    results = hrv_nonlinear(rri)
    assert results['sd2'] == pytest.approx(0.0002, 1e-4)

def test_hrv_nonlinear_sd12(rri_rpeaks) -> None:

    ''' Check for sd12 '''

    rri, _ = rri_rpeaks
    results = hrv_nonlinear(rri)
    assert results['sd12'] == pytest.approx(0.0002, 1e-4)

def test_hrv_nonlinear_sampen(rri_rpeaks) -> None:

    ''' Checks for sample entropy '''

    rri, _ = rri_rpeaks
    results = hrv_nonlinear(rri)
    assert results['sampen'] == pytest.approx(0.0002, 1e-4)

# Only valid for 1 hour recordings

# I am forcing here


### Still need to figure out how I am going to fit in the 1 hour recordings

def test_hrv_nonlinear_appen(rri_rpeaks) -> None:

    ''' Check for 1 hour 
    recordings '''

    rri, _ = rri_rpeaks
    results = hrv_nonlinear(rri)
    assert results['appen'] == pytest.approx(0.0002, 1e-4)


#### Testing auxilary for non - linear #######


def test_sampen_zeros(entropy_zeros) -> None:

    ''' Check the functions value for an
    expected output of input '''

    results = sample_entropy(entropy_zeros)
    assert results == pytest.approx(0, abs = 0.01)

def test_appen_zeros(entropy_zeros) -> None:

    ''' Check the functions value for an
    expected output of input '''

    results = approximate_entropy(entropy_zeros)
    assert results == pytest.approx(0, abs = 0.01)

def test_sampen_random(rand) -> None:
    
    ''' Check the functions value for an
    expected output of input '''
    
    results = sample_entropy(rand)
    assert results == pytest.approx(1, abs = 0.01)

def test_appen_random(rand) -> None:
    
    ''' Check the functions value for an
    expected output of input '''

    results = approximate_entropy(rand)
    assert results == pytest.approx(1, abs = 0.01)