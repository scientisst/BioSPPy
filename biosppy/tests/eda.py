import numpy as np
from matplotlib import pylab as plt
from ..features.eda import eda_features, eda_param


def test():
    eda_s = np.loadtxt("examples/eda.txt")  
    
    on, pks, amp, end = eda_param(eda_s, filt=True, size=1.5, sampling_rate=1000)
    for i in range(len(on)):
        print("t", type(on[i]))
        print("t", type(on))
        plt.figure()
        plt.plot(eda_s)
        plt.vlines(on[i], eda_s.min(), eda_s.max(), label="on", color="r")
        plt.vlines(pks[i], eda_s.min(), eda_s.max(), label="pks", color="b")
        plt.vlines(end[i], eda_s.min(), eda_s.max(), label="end", color="c")
        plt.hlines(eda_s[int(on[i])]+amp[i], 0, len(eda_s), label="amp", color="k")
        plt.legend()
        plt.show()

    args = feature_vector(eda_s, filt=True, size=1.5, sampling_rate=1000)

    for i in range(len(on)):
        plt.figure()
        plt.plot(eda_s)
        plt.vlines(args["onsets"][i], eda_s.min(), eda_s.max(), label="on", color="r")
        plt.vlines(args["peaks"][i], eda_s.min(), eda_s.max(), label="pks", color="b")
        plt.hlines(eda_s[int(args["onsets"][i])]+args["amps"][i], 0, len(eda_s), label="amp", color="k")
        
        plt.vlines(args["onsets"][i] + args["rise_ts"][i], eda_s.min(), eda_s.max(), label="pks", color="c")
        try:
            plt.vlines(args["onsets"][i] + args["half_rec"][i], eda_s.min(), eda_s.max(), label="half rec", color="pink")
        except:
            continue
        try:
            plt.vlines(args["onsets"][i] + args["six_rec"][i], eda_s.min(), eda_s.max(), label="six rec", color="orange")
        except:
            continue
        plt.legend()
        plt.show()





