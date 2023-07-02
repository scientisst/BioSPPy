import os
import sys
from biosppy import storage
import warnings
from biosppy.inter_plotting.event_annotator import event_annotator

warnings.simplefilter(action='ignore', category=FutureWarning)

# Setting current path
current_dir = os.path.dirname(sys.argv[0])

filenames = os.listdir(os.path.join('./examples'))

# Test platform for all signals except ACC
for fname in filenames:

    if fname != 'acc.txt':
        print(fname)
        signal, mdata = storage.load_txt(os.path.join('examples', fname))

        event_annotator(signal, mdata, window_size=6, window_stride=1.5,
                        annotations_dir=os.path.join(current_dir, 'examples'))
