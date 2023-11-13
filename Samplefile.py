import argparse

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import scipy as sp


def hdfig(subplots_def=None, scale=0.5):
    fig = plt.figure(figsize=(8, 4.5), dpi=scale * 1920 / 8)
    if subplots_def is None:
        return fig
    else:
        return fig, fig.subplots(*subplots_def)

def digitize(real_sequence: 'np.ndarray', nrbits: int = 8):
    result = np.rint(real_sequence).astype(int)
    maxpos = 2 ** (nrbits - 1) - 1
    maxneg = -(2 ** (nrbits - 1))
    result[result < maxneg] = maxneg
    result[result > maxpos] = maxpos
    return result


def simple_real_cross_power(ant_1_voltage, ant_2_voltage, nrbits: int = 4):
    # assert ant_1_voltage.shape == ant_2_voltage.shape
    s1 = ant_1_voltage #digitize(ant_1_voltage, nrbits=nrbits)
    s2 = ant_2_voltage #digitize(ant_2_voltage, nrbits=nrbits)
    return sp.signal.correlate(s1, s2, mode='same') / s1.shape[0]


def plot_real_xc(xc: 'np.ndarray[np.float]', width: int, sample_interval, caption=None):
    fig, ax = hdfig((1, 1))
    m = xc.shape[0] // 2
    lw = width // 2
    hw = width - lw
    delay = np.arange(-lw, hw) * sample_interval
    ax.plot(delay, xc[m - lw:m + hw])
    ax.set_xlabel('Delay [%s]' % delay.unit)
    if caption:
        ax.set_title(caption)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="calculating correlation as a in-field test",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("file1", type=str, help="Name of file 1")
    parser.add_argument("file2", type=str, help="Name of file 2")
    parser.add_argument("-w", '--window', type=int, help="Window of correlation plot")
    args = parser.parse_args()
    config = vars(args)

    file1 = np.fromfile(config['file1'], dtype=np.complex64)  # [0:3000000]
    file2 = np.fromfile(config['file2'], dtype=np.complex64)  # [0:3000000]

    delta_freq = 2.5 * u.MHz
    sample_interval = (1 / (2 * delta_freq)).to(u.s)

    if config['window'] is None:
        w = 100000
    else:
        w = config['window']

    try:
        corrcoeff = np.corrcoef(file1, file2)[0, 1]
        cross = simple_real_cross_power(file1 * 10, file2 * 10, nrbits=4)
    except ValueError:
        lendiff = len(file2)-len(file1)
        print(f"Sample loss has occurred between the two, the difference in lenght is: {np.abs(lendiff)}")
        print(f"Calculating the correlation where the two are of equal length")
        if lendiff<0:
            file1 = file1[:lendiff]
        else:
            file2 = file2[:-lendiff]
        corrcoeff = np.corrcoef(file1, file2)[0, 1]
        cross = simple_real_cross_power(file1 * 10, file2 * 10, nrbits=4)


    print(f"The crosscorrelation coefficient of the data is: {np.real(corrcoeff):.3f}")

    plot_real_xc(cross, w, sample_interval)


if __name__ == "__main__":
    main()