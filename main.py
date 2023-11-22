import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import scipy as sp
from scipy import signal

save_files = False
nrchan = 1024
measure_time = 5

sample_freq = 2.5*1e6

filename1 = "/disks/strw2/RadioAstronomy2023/Interferometer_Dataant1_1420.0mhz__27.10.202315.13.26.grc"
filename2 = "/disks/strw2/RadioAstronomy2023/Interferometer_Dataant2_1420.0mhz__27.10.202315.13.26.grc"
filename3 = "/disks/strw2/RadioAstronomy2023/ant3_1420.0mhz__27.10.202315.13.23.grc"
filename4 = "/disks/strw2/RadioAstronomy2023/ant4_1420.0mhz__27.10.202315.13.23.grc"
filename5 = "/disks/strw2/RadioAstronomy2023/ant5_1420.0mhz__27.10.202315.13.24.grc"
filename6 = "/disks/strw2/RadioAstronomy2023/ant6_1420.0mhz__27.10.202315.13.24.grc"
filename7 = "/disks/strw2/RadioAstronomy2023/dataant7_1420.0mhz__27.10.202315.13.25.grc"
filename8 = "/disks/strw2/RadioAstronomy2023/dataant8_1420.0mhz__27.10.202315.13.25.grc"

nr_samples = int(measure_time*sample_freq - measure_time*sample_freq%nrchan)
x = np.arange(nr_samples)/sample_freq

data1 = np.fromfile(filename1, dtype=np.complex64, count=nr_samples)
data2 = np.fromfile(filename2, dtype=np.complex64, count=nr_samples)


startdata = data1[:200000]
startx = x[:200000]
plt.plot(startx, startdata)
plt.title("Raw data with artifact")
plt.xlabel("seconds")
plt.savefig("reportplots/rawdata", dpi=300)
plt.show()

#base offset is needed to filter out weird effect at the start of each signal
base_offset = 40000 #in nr samples
print(f"Base offset is {base_offset/sample_freq} seconds")


#Take some part of antenna 1 and a small part of antenna 2 and correlate antenna 2 across antenna 1
delaysize1 = 200000
delaysize2 = 100000
delaydata1 = data1[base_offset:base_offset + delaysize1]
delaydata2 = data2[base_offset:base_offset + delaysize2]
delayx1 = x[base_offset:base_offset + delaysize1]
# delayx2 = x[base_offset:base_offset + delaysize2]

plt.plot(delayx1, delaydata1)
plt.title(f"Raw data example shifted by {base_offset / sample_freq:.3f} seconds to remove artifact")
plt.xlabel("Seconds")
plt.savefig("reportplots/rawdatashifted", dpi=300)
plt.show()


print(f"Find delay using {delaysize1/sample_freq} of antenna 1 and {delaysize2/sample_freq} of antenna 2")

corr = signal.correlate(delaydata1, delaydata2, mode="valid") #mode valid means no zero padding at the beginning and end
corrx = x[base_offset:base_offset + delaysize1 - delaysize2 + 1]

plt.plot(corrx, np.real(corr))
plt.title("Real part of correlation")
plt.xlabel("Seconds")
plt.savefig("reportplots/realcorr", dpi=300)
plt.show()

plt.plot(corrx, np.imag(corr))
plt.title("Imaginary part of correlation")
plt.xlabel("Seconds")
plt.savefig("reportplots/imagcorr", dpi=300)
plt.show()

plt.plot(corrx, np.real(corr)**2 + np.imag(corr)**2)
plt.title(f"Real(corr)$^2$ + Imag(corr)$^2$")
plt.xlabel("Seconds")
plt.savefig("reportplots/magcorr", dpi=300)
plt.show()

delay_samples = np.argmax(np.real(corr)**2 + np.imag(corr)**2)
print(f"Delay found (after base delay): {delay_samples} samples or {delay_samples/sample_freq} seconds")

delaydata1 = data1[base_offset + delay_samples:base_offset + delaysize1 + delay_samples]
delaydata2 = data2[base_offset:base_offset + delaysize2]
corr = signal.correlate(delaydata1, delaydata2, mode="valid")
plt.plot(corrx, np.real(corr)**2 + np.imag(corr)**2)
plt.title(f"Real(corr)$^2$ + Imag(corr)$^2$, shifted")
plt.xlabel("Seconds")
plt.savefig("reportplots/magcorrshifted", dpi=300)
plt.show()


print(f"FFT window: {nrchan} samples or {nrchan/sample_freq} seconds")
print("measure samples: ", nr_samples, "total of ", nr_samples/(2.5*1e6), "seconds")

nrwindows = int((nr_samples-base_offset-delay_samples) / nrchan)

data1 = data1[base_offset+delay_samples:base_offset+delay_samples+nrwindows*nrchan].reshape(nrwindows, nrchan)
data2 = data2[base_offset:base_offset+nrwindows*nrchan].reshape(nrwindows, nrchan)
x = x[base_offset:base_offset+nrchan]

FFT1 = np.fft.fft(data1, axis=1)
FFT2 = np.fft.fft(data2, axis=1)

freqs = np.fft.fftfreq(len(x), d=1e6/sample_freq)

plt.plot(freqs, np.real(FFT1[0]))
plt.title("Real part of FFT not FFTshifted")
plt.xlabel("Frequency (MHz)")
plt.savefig("reportplots/realFFTnonshifted", dpi=300)
plt.show()


FFT1 = np.fft.fftshift(FFT1)
FFT2 = np.fft.fftshift(FFT2)
freqs = np.fft.fftshift(freqs)


plt.plot(freqs, np.real(FFT1[0]))
plt.title("Real part of FFT, shifted")
plt.xlabel("Frequency (MHz)")
plt.savefig("reportplots/realFFT", dpi=300)
plt.show()

crosscorr = np.mean(FFT1 * np.conj(FFT2), axis=0)

plt.plot(freqs, np.absolute(crosscorr))
plt.title(f"Magnitude of cross correlation of FFTs averaged over {measure_time} seconds, including clock peak")
plt.savefig("reportplots/magcrosscorrwithpeak", dpi=300)
plt.show()

interpolationrange = 1
peak = np.argmax(np.abs(crosscorr))
left_index = peak - interpolationrange
right_index = peak + interpolationrange
crosscorr[left_index:right_index] = np.linspace(crosscorr[left_index], crosscorr[right_index], 2*interpolationrange)

plt.plot(freqs, np.absolute(np.real(crosscorr)))
plt.title(f"Real part of cross correlation of FFTs averaged over {measure_time} seconds")
plt.savefig("reportplots/realcrosscorr", dpi=300)
plt.show()

plt.plot(freqs, np.angle(crosscorr))
plt.title(f"Imaginary part of cross correlation of FFTs averaged over {measure_time} seconds")
plt.savefig("reportplots/imagcrosscorr", dpi=300)
avg_range = int(nrchan/4)
avg_angle = np.average(np.angle(crosscorr)[avg_range:-avg_range])
plt.hlines(avg_angle, freqs[avg_range], freqs[-avg_range], colors="red")
plt.show()

print("line at: ", avg_angle)
