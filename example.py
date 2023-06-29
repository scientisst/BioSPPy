import os
import sys

from biosppy import storage
import numpy as np
import warnings

from biosppy.signals import ecg
from biosppy.signals.acc import acc
from biosppy.signals.eda import eda
from biosppy.signals.ppg import ppg
from biosppy.signals.ecg import ecg


warnings.simplefilter(action='ignore', category=FutureWarning)

# load raw ECG and ACC signals
ecg_signal, _ = storage.load_txt('./examples/ecg.txt')
acc_signal, _ = storage.load_txt('./examples/acc.txt')
ppg_signal, _ = storage.load_txt('./examples/ppg.txt')
eda_signal, _ = storage.load_txt('./examples/eda.txt')

# Setting current path
current_dir = os.path.dirname(sys.argv[0])
ecg_plot_path = os.path.join(current_dir, 'ecg.png')
acc_plot_path = os.path.join(current_dir, 'acc.png')
# eda_plot_path = os.path.join(current_dir, 'eda.png')

# Process it and plot. Set interactive=True to display an interactive window
out_ecg = ecg(signal=ecg_signal, sampling_rate=1000., show=True, interactive=True)

out_acc = acc(signal=acc_signal, sampling_rate=1000., show=True, interactive=True)
# filt = out_ppg["filtered"]
# # np.savetxt("ppg_filt.csv", filt, delimiter=",")

# out_ecg = ecg.ecg(signal=ecg_signal, sampling_rate=1000., path=ecg_plot_path, interactive=True)
# out_acc = acc(signal=acc_signal, sampling_rate=1000., path=acc_plot_path, interactive=True)
# out_ppg = ppg(signal=ppg_signal, sampling_rate=1000.)
# out_eda = eda(signal=eda_signal, sampling_rate=1000.)
