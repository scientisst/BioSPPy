import os
import sys

from biosppy import storage

import warnings

from biosppy.inter_plotting.event_annotator import event_annotator
from biosppy.signals import ecg
from biosppy.signals.acc import acc

warnings.simplefilter(action='ignore', category=FutureWarning)

# load raw ECG and ACC signals
acc_signal, mdata = storage.load_txt('./examples/acc.txt')
ecg_signal, ecg_mdata = storage.load_txt('./examples/ecg.txt')


# Setting current path
current_dir = os.path.dirname(sys.argv[0])
acc_plot_path = os.path.join(current_dir, 'acc.png')

# Process it and plot. Set interactive=True to display an interactive window
out_acc = acc(signal=acc_signal, sampling_rate=1000., show=False, interactive=False)

print(mdata)
print(ecg_mdata)
print(acc_signal.shape)

event_annotator(acc_signal, mdata, 6, 1.5, path_to_signal=None)