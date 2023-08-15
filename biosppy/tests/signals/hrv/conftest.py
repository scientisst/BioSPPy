import pytest
import pandas as pd
import numpy as np
import os


@pytest.fixture(scope='module', params=[
    file_name
    for directory in ['Data/5_min_segments', 'Data/1_hour_segments']
    for file_name in os.listdir(directory)
    if file_name.endswith('.csv') and file_name in pd.read_csv('Data/KubiosResults/OneShortOneLong.csv', skiprows=2).iloc[:, 0].tolist()
])
def get_file_data(request):
    
    csv_file_path = 'Data/OneShortOneLong.csv'
    df = pd.read_csv(csv_file_path, skiprows=2)

    file_name = request.param

    folder_paths = ['Data/5_min_segments', 'Data/1_hour_segments']
    file_path = None
    for folder_path in folder_paths:
        if os.path.exists(os.path.join(folder_path, file_name)):
            file_path = os.path.join(folder_path, file_name)
            break

    column_names = ['rrinterval', 'BPM', 'Time']
    file_df = pd.read_csv(file_path, names=column_names)
    arr_rri = file_df['rrinterval'].values * 1000
    rri = np.array(arr_rri)
    arr_rpeaks = file_df['Time'].values * 1000
    rpeaks = np.array(arr_rpeaks)
    result_df_row = df[df.iloc[:, 0] == file_name]

    return rri, rpeaks, result_df_row




@pytest.fixture
def dummy():
    fake = np.array([1.0,2.0,3.0,2500.0])
    return fake

@pytest.fixture
def fbands():
    frequency_bands = {'LF': (0.04, 0.15), 'HF': (0.15, 0.4)}
    return frequency_bands

@pytest.fixture
def entropy_zeros():
    zeros = np.zeros(500)
    return zeros

@pytest.fixture
def rand():
    random = np.random.rand(500)
    return random