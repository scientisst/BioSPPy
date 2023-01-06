# Imports
# 3rd party
import numpy as np
from matplotlib import pylab as plt
from scipy import interpolate
from PIL import Image
from matplotlib import cm

# local
from biosppy.signals.eda import eda, eda_events, eda_features, kbk_scr, edr, edl
from biosppy.signals import tools
from biosppy.features import phase_space, time, frequency, cepstral, time_freq
import biosppy 

# load data
signal, eda_metadata = biosppy.storage.load_txt("examples/eda.txt")
sampling_rate = eda_metadata["sampling_rate"]
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
#plt.show()
plt.close()

# spectrogram
#window = np.hamming(len(signal))
#wind_signal = signal * window
#
#spectrum = np.abs(np.fft.fft(wind_signal, len(wind_signal)))
#
#print(len(wind_signal), len(spectrum))
#f = np.nan_to_num(np.array(np.fft.fftfreq(len(spectrum))))
#spectrum = np.nan_to_num(spectrum[:len(spectrum)//2])
#spectrum /= len(spectrum)
#f = np.abs(f[:len(f)//2]*sampling_rate)

freqs, power = biosppy.signals.tools.power_spectrum(signal)

# plot spectrogram
plt.figure(dpi=300)
plt.plot(freqs, power, label="Raw Signal Spectrogram")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.xlim(0, 100)
plt.savefig("raw_spectrogram.pdf", bbox_inches='tight', format="pdf")
#plt.show()
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
#plt.show()
plt.close()

freqs, power = biosppy.signals.tools.power_spectrum(filtered)
#window = np.hamming(len(filtered))
#wind_signal = filtered * window
#spectrum = np.abs(np.fft.fft(wind_signal, len(wind_signal)))
#
#print(len(wind_signal), len(spectrum))
#f = np.nan_to_num(np.array(np.fft.fftfreq(len(spectrum))))
#spectrum = np.nan_to_num(spectrum[:len(spectrum)//2])
#spectrum /= len(spectrum)
#f = np.abs(f[:len(f)//2]*sampling_rate)

# plot spectrogram
plt.figure(dpi=300)
plt.plot(freqs, power, label="Filtered Signal Spectrogram")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 100)
plt.legend()
plt.savefig("filtered_spectrogram.pdf", bbox_inches='tight', format="pdf")
#plt.show()
plt.close()


# get eda characteristic events
onsets, peaks, amps, _ = eda_events(filtered, filt=True, size=0.9, sampling_rate=1000)
args = eda_features(filtered, filt=True, size=0.9, sampling_rate=1000)

plt.figure(dpi=300)
plt.plot(filtered, label="Filtered Signal")

for i in range(len(onsets)):
    plt.plot(args["onsets"][i], filtered[args["onsets"][i]],  ".", label="onsets", color="r")
    plt.plot(args["peaks"][i], filtered[args["peaks"][i]],  "X", label="peaks", color="b")
    #plt.hlines(signal[int(args["onsets"][i])]+args["amps"][i], 0, len(signal), label="amplitudes", color="k")
    
    #plt.vlines(args["onsets"][i] + args["rise_ts"][i], signal.min(), signal.max(), label="rise time check", color="c")
    try:
        plt.plot(args["onsets"][i] + args["half_rec"][i], filtered[args["onsets"][i] + args["half_rec"][i]], "^", label="half recovery time", color="pink")
    except Exception as e:
        print(e)
        continue
    try:
        plt.plot(args["onsets"][i] + args["six_rec"][i], filtered[args["onsets"][i] + args["six_rec"][i]], "o", label="six recovery time", color="orange")
    except Exception as e:
        print(e)
        continue
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.savefig("eda_events.pdf", bbox_inches='tight', format="pdf")
plt.show()
plt.close()

# eda decomposition
edr = edr(filtered, sampling_rate=sampling_rate)["edr"]
edl = edl(filtered, sampling_rate=sampling_rate)["edl"]
# or

#edl_on = np.hstack((ts[0], ts[onsets], ts[-1]))
#edl_amp = np.hstack((filtered[0], filtered[onsets], filtered[-1]))
#f = interpolate.interp1d(edl_on, edl_amp)
#edl_n = f(ts)

fig, ax1 = plt.subplots(dpi=300)
ax1.plot(ts, filtered, label="Filtered EDA")
#plt.plot(ts[1:], edr, label="EDR")
ax1.plot(ts, edl, label="EDL")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")

ax2 = ax1.twinx()
ax2.plot(ts[1:], edr, label="EDR", color="g")

fig.legend(loc="upper left")
plt.savefig("eda_decomposition.pdf", bbox_inches='tight', format="pdf")
#plt.show()
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
#plt.show()
plt.close('all')

# summary
onsets, filtered, edr, edl, onsets, peaks, amplitudes = eda(signal, path="eda", show="True")

rp_features = phase_space.phase_space_features(filtered)

print(rp_features)
print(len(rp_features))

time_features = time.time_features(filtered, sampling_rate)

print("time f", time_features)
print("time f", len(time_features))

cepstral_features = cepstral.cepstral_features(filtered, sampling_rate)

print("cepstral f", cepstral_features)
print("cepstral f", len(cepstral_features))

time_freq_features = time_freq.time_freq_features(filtered, sampling_rate)

print("time_freq_features f", time_freq_features)
print("time_freq_features f", len(time_freq_features))

freq_features = frequency.freq_features(filtered, sampling_rate)

print("freq_features f", freq_features)
print("freq_features f", len(freq_features))
