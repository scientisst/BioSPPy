from biosppy.signals.hrv import compute_rri
from biosppy.signals.hrv import rri_filter
from biosppy.signals.hrv import hrv
from biosppy.signals.hrv import detrend_window
from biosppy.signals.hrv import hrv_frequencydomain
import pandas as pd
import numpy as np
import pytest
import os

file_path = '/Users/kingsize/Documents/4_Ano/segundo_semestre/PIC/Project/all_results/Results/5_min_segments/16420_300_600.csv'

column_names = ['rrinterval', 'BPM', 'Time']

df = pd.read_csv(file_path, names=column_names)

new_df = df['Time'].astype(float)*1000

new_df = new_df.values*1000

arr = df['rrinterval'].values*1000

#arr = arr.astype(float)

#lista = [0.04,0.15]


results = hrv_frequencydomain(arr, features_only = False)
print(results.keys)

# Testing time domain features


## Todas as funções que envolvem o rri filter não 
## estão corretas porque envolvem um threshold muito baixo

def test_hrv_empty() -> None:
    
    ''' Test what happens when RRI 
    and RPeaks are not provided. '''

    with pytest.raises(TypeError):
        hrv()

def test_hrv_parameters_input(rri_rpeaks):

    ''' We provide a valid input, 
    with a wrong parameter.
    We expect error. '''

    _,rpeaks = rri_rpeaks
    with pytest.raises(ValueError):
        hrv(rpeaks=rpeaks, parameters='invalid')

def test_hrv_sampling_period(rri_rpeaks):

    ''' We provide a non - float 
    sampling rate to check'''

    _,rpeaks = rri_rpeaks
    with pytest.raises(ValueError):
        hrv(rpeaks = rpeaks, sampling_rate='invalid')

def test_hrv_rpeaks() -> None:

    ''' Test if the function raises
    an error if rpeaks is invalid'''

    with pytest.raises(ValueError):
        hrv(rpeaks = 'invalid')

def test_hrv_rpeaks_empty() -> None:

    ''' Test if how the function 
    behaves for an empty list '''

    with pytest.raises(ValueError):
        hrv(rpeaks = [])

def test_hrv_compute_rri(dummy):

    ''' Check if comput rri 
    raises a warning when 
    provided through hrv '''

    with pytest.warns(UserWarning, match="RR-intervals appear to be out of normal parameters"):
        compute_rri(dummy)

def test_hrv_rri_provided(rri_rpeaks):

    ''' Providing  an rri and check
    if it gives out the same '''

    rri,_ = rri_rpeaks
    results = hrv(rri = rri, detrend_rri=False)
    assert np.array_equal(results['rri'], rri)

def test_hrv_rri_not_provided(rri_rpeaks):

    ''' Providing rpeaks, and understand 
    if it gives out what is suppose to '''

    rri, rpeaks = rri_rpeaks
    results = hrv(rpeaks=rpeaks, detrend_rri=False)
    assert np.array_equal(results['rri'], rri[1:])

def test_hrv_rri_empty():

    ''' We provide an invalid rri to 
    understand if the function correctly
    raises an error. This error is a 
    consequence of the np.sum(rri) '''

    with pytest.raises(ValueError):
        hrv(rri = [])

def test_hrv_rri_invalid():

    ''' We provide an invalid datatype
    to check if the function raises the 
    error '''

    with pytest.raises(TypeError):
        hrv(rri = 'invalid')

def test_hrv_features_only(rri_rpeaks):

    ''' We check if the function 
    returns only the features 
    when features_only = True '''

    rri, _ = rri_rpeaks
    results = hrv(rri = rri, detrend_rri = False, features_only=True)
    assert 'rri' not in results.keys()

def test_hrv_not_features_only(rri_rpeaks):

    ''' We check if the function 
    returns also the rri when it 
    is features_only = True '''

    _, rpeaks = rri_rpeaks
    results = hrv(rpeaks = rpeaks, detrend_rri = False, features_only = False)
    assert 'rri' in results.keys()

## Maybe to try to test when parameters = all, don't know how to do it ##

def test_hrv_paramenters_all(rri_rpeaks):

    ''' Check if a time domain, 
    frequency domain and non linear 
    domain were calculated to see if 
    np.duratio == np.inf'''

    rri, _ = rri_rpeaks
    results = hrv(rri = rri, parameters = 'all', detrend_rri = False, features_only = False)
   
    print(results.keys())

    required_keys = [
        'rri', 'hr', 'hr_min', 'hr_max', 'hr_minmax', 'hr_mean', 
        'hr_median', 'rr_min', 'rr_max', 'rr_minmax', 'rr_mean', 
        'rr_median', 'rmssd', 'nn50', 'pnn50', 'sdnn', 'hti', 
        'tinn', 'ulf_pwr', 'ulf_peak', 'ulf_rpwr', 'vlf_pwr', 
        'vlf_peak', 'vlf_rpwr', 'lf_pwr', 'lf_peak', 'lf_rpwr',
        'hf_pwr', 'hf_peak', 'hf_rpwr', 'vhf_pwr', 'vhf_peak', 
        'vhf_rpwr', 'lf_hf', 'lf_nu', 'hf_nu', 's', 'sd1', 'sd2', 
        'sd12', 'sd21', 'sampen', 'appen']

    assert all(key in results.keys() for key in required_keys)

## Check is errors were raised ##

def test_hrv_timedomain_value_error(dummy, capsys) -> None:

    ''' We provide an invalid input,
    but with parameters = time.
    We expect error. '''

    hrv(rri=dummy, parameters='time')
    captured = capsys.readouterr()
    assert captured.out.strip() == 'Time-domain features not computed. Check input.'

def test_hrv_frequencydomain_value_error(dummy, capsys) -> None:

    ''' We provide an invalid input,
    but with parameters = frequency.
    We expect error. '''

    hrv(rri=dummy, parameters='frequency')
    captured = capsys.readouterr()
    assert captured.out.strip() == 'Frequency-domain features not computed. Check input.'

def test_hrv_nonlineardomain(dummy, capsys) -> None:

    ''' We provide an invalid input,
    but with parameters = nonlinear.
    We expect error. '''

    hrv(rri=dummy, parameters='non-linear')
    captured = capsys.readouterr()
    assert captured.out.strip() == 'Non-linear features not computed. Check input.'


###### Testar ainda para o output ######

# Perguntar na reunião, qual o sampling rate


#### Testar o Time Domain ####


###### Compute rri ######


def test_compute_rri(rri_rpeaks) -> None:

    ''' We are verifying the correctness
    output of this auxiliary function '''

    rri, rpeaks = rri_rpeaks
    result = compute_rri(rpeaks)
    assert np.array_equal(result, rri[1:])

def test_compute_rri_warning(dummy):

    ''' Veryfing a warning is raised when
    provided with a wrong input '''
    
    with pytest.warns(UserWarning, match="RR-intervals appear to be out of normal parameters"):
        compute_rri(dummy)

def test_rri_filter_not_needed(rri_rpeaks) -> None:

    ''' Veryfing if the filteration is
    well performed with non needed filteration '''

    rri, _ = rri_rpeaks
    print(rri)
    filtered = rri_filter(rri=rri, threshold=2000)
    assert np.array_equal(rri, filtered)


def test_rri_filter(dummy):
    filtered = rri_filter(rri = dummy, threshold=2000)
    assert not np.array_equal(filtered, dummy)


####### Detrend Window #######


def test_detrend_window_output_shape(rri_rpeaks):

    """ Test if the output shape of 
    the detrend_window function 
    is as expected """

    rri, _ = rri_rpeaks

    # Call the function
    rri_det = detrend_window(rri, win_len=2000)

    # Check if the output has the expected shape
    assert len(rri_det) == len(rri)

def test_detrend_window_bigger(rri_rpeaks):

    ''' Test what happens to the 
    function when the detrend
    window length is bigger'''

    rri, _ = rri_rpeaks
    rri_det = detrend_window(rri, win_len = 100)
    print(rri)
    print(rri_det)
    assert len(rri_det) == len(rri)

def test_detrend_window_smoothing_factor(rri_rpeaks):

    """ Test if the smoothing_factor 
    parameter works as expected """

    rri, _ = rri_rpeaks

    # Call the function with a custom smoothing factor
    rri_det = detrend_window(rri, win_len=2000, smoothing_factor=1000)

    # Check if the output has the expected shape
    assert len(rri_det) == len(rri)

def test_detrend_window_invalid_input():

    """ Test if the function raises an error 
    when invalid input is given """

    # Call the function with an invalid input type
    invalid_input = 'invalid_input'
    with pytest.raises(TypeError):
        detrend_window(invalid_input, win_len=2000) 

def test_detrend_window_invalid_window(rri_rpeaks):

    ''' Test if the function gives out an error
    when an invalid window is provided'''

    rri, _ = rri_rpeaks

    window = 'Invalid Window'
    with pytest.raises(ValueError):
        detrend_window(rri, win_len = window)
