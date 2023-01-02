# Imports
# 3rd party
import numpy as np
from matplotlib import pylab as plt
from scipy import interpolate
from PIL import Image
from matplotlib import cm

# local
from biosppy.signals.eda import eda, eda_param, eda_features, kbk_scr
from biosppy.signals import tools
from biosppy.features import phase_space

# load data
signal = np.array(np.loadtxt("examples/eda.txt"))
sampling_rate = 1000.

# get time vectors
length = len(signal)
T = (length - 1) / sampling_rate
ts = np.linspace(0, T, length, endpoint=True)

# plot raw data
plt.figure(dpi=300)
plt.plot(ts, signal, label="raw signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.legend()
plt.savefig("raw_data.pdf", bbox_inches='tight', format="pdf")
plt.show()
plt.close()

# spectrogram
window = np.hamming(len(signal))
wind_signal = signal * window

spectrum = np.abs(np.fft.fft(wind_signal, len(wind_signal)))

print(len(wind_signal), len(spectrum))
f = np.nan_to_num(np.array(np.fft.fftfreq(len(spectrum))))
spectrum = np.nan_to_num(spectrum[:len(spectrum)//2])
spectrum /= len(spectrum)
f = np.abs(f[:len(f)//2]*sampling_rate)

# plot spectrogram
plt.figure(dpi=300)
plt.plot(f, spectrum, label="Raw Signal Spectrogram")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig("raw_spectrogram.pdf", bbox_inches='tight', format="pdf")
plt.show()
plt.close()
# zoom at 50hz we observe a AC peak

# remove noise
filtered, _, _ = tools.filter_signal(
        signal=signal,
        ftype="butter",
        band="lowpass",
        order=4,
        frequency=5,
        sampling_rate=sampling_rate)


# plot raw and filtered data
plt.figure(dpi=300)
plt.plot(ts, signal, label="Raw Signal")
plt.plot(ts, filtered, label="Filterd Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.legend()
plt.savefig("filtered_data.pdf", bbox_inches='tight', format="pdf")
plt.show()
plt.close()

window = np.hamming(len(filtered))
wind_signal = filtered * window
spectrum = np.abs(np.fft.fft(wind_signal, len(wind_signal)))

print(len(wind_signal), len(spectrum))
f = np.nan_to_num(np.array(np.fft.fftfreq(len(spectrum))))
spectrum = np.nan_to_num(spectrum[:len(spectrum)//2])
spectrum /= len(spectrum)
f = np.abs(f[:len(f)//2]*sampling_rate)

# plot spectrogram
plt.figure(dpi=300)
plt.plot(f, spectrum, label="Filtered Signal Spectrogram")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig("filtered_spectrogram.pdf", bbox_inches='tight', format="pdf")
plt.show()
plt.close()


# get eda characteristic events
onsets, peaks, amps, _ = eda_param(filtered, filt=True, size=0.9, sampling_rate=1000)
args = eda_features(filtered, filt=True, size=0.9, sampling_rate=1000)

plt.figure(dpi=300)
plt.plot(filtered, label="Filtered Signal")

for i in range(len(onsets)):
    plt.vlines(args["onsets"][i], signal.min(), signal.max(), label="onsets", color="r")
    plt.vlines(args["peaks"][i], signal.min(), signal.max(), label="peaks", color="b")
    #plt.hlines(signal[int(args["onsets"][i])]+args["amps"][i], 0, len(signal), label="amplitudes", color="k")
    
    #plt.vlines(args["onsets"][i] + args["rise_ts"][i], signal.min(), signal.max(), label="rise time check", color="c")
    try:
        plt.vlines(args["onsets"][i] + args["half_rec"][i], signal.min(), signal.max(), label="half recovery time", color="pink")
    except Exception as e:
        print(e)
        continue
    try:
        plt.vlines(args["onsets"][i] + args["six_rec"][i], signal.min(), signal.max(), label="six recovery time", color="orange")
    except Exception as e:
        print(e)
        continue
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.savefig("eda_events.pdf", bbox_inches='tight', format="pdf")
plt.show()
plt.close()

# eda decomposition

# differentiation
df = np.diff(signal)

# smooth
size = int(1.0 * sampling_rate)
edr, _ = tools.smoother(signal=df, kernel="bartlett", size=size, mirror=True)

# smooth
size = int(10.0 * sampling_rate)
edl, _ = tools.smoother(signal=filtered, kernel="bartlett", size=size, mirror=True)

# or

edl_on = np.hstack((ts[0], ts[onsets], ts[-1]))
edl_amp = np.hstack((filtered[0], filtered[onsets], filtered[-1]))
f = interpolate.interp1d(edl_on, edl_amp)
edl_n = f(ts)

fig, ax1 = plt.subplots(dpi=300)
ax1.plot(ts, filtered, label="Filtered EDA")
#plt.plot(ts[1:], edr, label="EDR")
ax1.plot(ts, edl_n, label="EDL")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")

ax2 = ax1.twinx()
ax2.plot(ts[1:], edr, label="EDR", color="g")

fig.legend()
plt.savefig("eda_decomposition.pdf", bbox_inches='tight', format="pdf")
plt.show()
plt.close()

# recurrence plot
rec = phase_space.rec_plot(filtered)["rec_plot"]
rec = (rec - np.min(rec)) / (np.max(rec) - np.min(rec))
image_rec = Image.fromarray(np.uint8(cm.viridis.reversed()(rec)*255))
image_rec = image_rec.resize((224, 224))

plt.figure(dpi=300)
plt.imshow(image_rec)
plt.grid(b=None)
plt.ylabel("Resampled Sample")
plt.xlabel("Resampled Sample")
plt.savefig("recurrence_plot.pdf", bbox_inches='tight', format="pdf")
plt.show()
plt.close('all')

# summary
eda(signal)


