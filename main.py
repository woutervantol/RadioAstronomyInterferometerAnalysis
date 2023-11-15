import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import scipy as sp
from scipy import signal


delta_freq = 2.5 * u.MHz
sample_interval = (1 / (2 * delta_freq)).to(u.s)

filename1 = "/disks/strw2/RadioAstronomy2023/Interferometer_Dataant1_1420.0mhz__27.10.202315.13.26.grc"
filename2 = "/disks/strw2/RadioAstronomy2023/Interferometer_Dataant2_1420.0mhz__27.10.202315.13.26.grc"


wrongdata = np.fromfile(filename1, dtype=np.complex64, count=200000, offset=0)
plt.plot(np.arange(len(wrongdata)) / (2.5*1e6), wrongdata)
plt.title("Raw data with artifact")
plt.xlabel("seconds")
plt.show()

#base offset is needed to filter out weird effect at the start of each signal
base_offset = 2**17 #in nr samples
print(f"Base offset is {base_offset / (2.5*1e6)} seconds")

#Take 0.1 seconds of antenna 1 and a small part of antenna 2 and correlate antenna 2 across antenna 1
rawdata1 = np.fromfile(filename1, dtype=np.complex64, count=200000, offset=base_offset*8)
rawdata2 = np.fromfile(filename2, dtype=np.complex64, count=100000, offset=base_offset*8)

plt.plot(np.arange(len(rawdata1)) / (2.5*1e6), rawdata1)
plt.title(f"Raw data example shifted by {base_offset / (2.5*1e6):.3f} seconds to remove artifact")
plt.xlabel("Seconds")
plt.show()


corr = signal.correlate(rawdata1, rawdata2, mode="valid") #mode valid means no zero padding at the beginning and end

plt.plot(signal.correlation_lags(len(rawdata1), len(rawdata2), mode="valid") / (2.5*1e6), np.real(corr))
plt.title("Real part of correlation")
plt.xlabel("Seconds")
plt.show()

plt.plot(signal.correlation_lags(len(rawdata1), len(rawdata2), mode="valid") / (2.5*1e6), np.imag(corr))
plt.title("Imaginary part of correlation")
plt.xlabel("Seconds")
plt.show()

plt.plot(signal.correlation_lags(len(rawdata1), len(rawdata2), mode="valid") / (2.5*1e6), np.real(corr)**2 + np.imag(corr)**2)
plt.title(f"Real(corr)$^2$ + Imag(corr)$^2$")
plt.xlabel("Seconds")
plt.show()


delay_samples = np.argmax(np.real(corr)**2 + np.imag(corr)**2)
print(f"Delay found (after base delay): {delay_samples} samples or {delay_samples /  (2.5*1e6)} seconds")


nrchan = 2**15#int(2**16)
t_in_chan = nrchan / (2.5*1e6)
print(f"FFT window: {nrchan} samples or {t_in_chan} seconds")

#make the total number of samples a multiple of the size of the FFT window
measure_time = 30 #seconds
nr_samples = int(measure_time*(2.5*1e6) - measure_time*(2.5*1e6)%nrchan)

data1_5sec = np.fromfile(filename1, dtype=np.complex64, count=nr_samples, offset=base_offset*8 + delay_samples*8)
data2_5sec = np.fromfile(filename1, dtype=np.complex64, count=nr_samples, offset=base_offset*8)

data1_5sec = data1_5sec.reshape(int(nr_samples/nrchan), nrchan)
data2_5sec = data2_5sec.reshape(int(nr_samples/nrchan), nrchan)

FFT1 = np.fft.fft(data1_5sec, axis=1)
FFT2 = np.fft.fft(data2_5sec, axis=1)

plt.plot(range(len(FFT1[0])), np.real(FFT1[0]))
plt.title("Real part of FFT not FFTshifted")
plt.show()

FFT1 = np.fft.fftshift(FFT1)
FFT2 = np.fft.fftshift(FFT2)

freqs = np.fft.fftshift(np.fft.fftfreq(len(FFT1[0])))


plt.plot(freqs, np.real(FFT1[0]))
plt.title("Real part of FFT, shifted")
plt.show()


# plt.plot(freqs, np.real(FFT1[0]))
# plt.title("Real part of FFT of single window of data1 without clock peak")
# plt.show()
# plt.plot(freqs, np.imag(FFT1[0]))
# plt.title("Imaginary part of FFT of single window of data1 without clock peak")
# plt.show()
# plt.plot(freqs, np.real(FFT2[0]))
# plt.title("Real part of FFT of single window of data2 after removing clock peak")
# plt.show()
# plt.plot(freqs, np.imag(FFT2[0]))
# plt.title("Imaginary part of FFT of single window of data2 without clock peak")
# plt.show()



autocorrelation = np.mean(FFT1 * np.conj(FFT1), axis=0)

# interpolate the clock peak
iprange = 20
left_index = np.argmax(autocorrelation)-iprange
right_index = np.argmax(autocorrelation)+iprange
autocorrelation[left_index:right_index] = np.linspace(autocorrelation[left_index], autocorrelation[right_index], 2*iprange)

plt.plot(np.linspace(1420 - 1.25, 1420 + 1.25, len(autocorrelation)), np.real(autocorrelation))
plt.title("Autocorrelation test real (should be like assignment 1?)")
plt.show()
plt.plot(np.linspace(1420 - 1.25, 1420 + 1.25, len(autocorrelation)), np.imag(autocorrelation))
plt.title("Autocorrelation test imag (should be like assignment 1?)")
plt.show()

crosscorr = np.mean(FFT1 * np.conj(FFT2), axis=0)
# plt.plot(np.linspace(1420 - 1.25, 1420 + 1.25, len(crosscorr)), np.real(crosscorr))
# plt.title(f"Real part of cross correlation of FFTs averaged over {measure_time} seconds")
# plt.show()
# plt.plot(np.linspace(1420 - 1.25, 1420 + 1.25, len(crosscorr)), np.imag(crosscorr))
# plt.title(f"Imaginary part of cross correlation of FFTs averaged over {measure_time} seconds")
# plt.show()
plt.plot(np.linspace(1420 - 1.25, 1420 + 1.25, len(crosscorr)), np.abs(crosscorr))
plt.title(f"Magnitude of cross correlation of FFTs averaged over {measure_time} seconds")
plt.show()


iprange = 30
left_index = np.argmax(np.abs(crosscorr))-iprange
right_index = np.argmax(np.abs(crosscorr))+iprange
crosscorr[left_index:right_index] = np.linspace(crosscorr[left_index], crosscorr[right_index], 2*iprange)

plt.plot(np.linspace(1420 - 1.25, 1420 + 1.25, len(crosscorr)), np.real(crosscorr))
plt.title(f"Real part of cross correlation of FFTs averaged over {measure_time} seconds")
plt.show()
plt.plot(np.linspace(1420 - 1.25, 1420 + 1.25, len(crosscorr)), np.imag(crosscorr))
plt.title(f"Imaginary part of cross correlation of FFTs averaged over {measure_time} seconds")
plt.show()
plt.plot(np.linspace(1420 - 1.25, 1420 + 1.25, len(crosscorr)), np.abs(crosscorr))
plt.title(f"Magnitude of cross correlation of FFTs averaged over {measure_time} seconds")
plt.show()