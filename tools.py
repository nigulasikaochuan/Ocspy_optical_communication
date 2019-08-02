from scipy.constants import c
from scipy.signal import welch

import Signal
import numpy as np
import matplotlib.pyplot as plt


def upsample(symbol_x, sps):
    '''

    :param symbol_x: 1d array
    :param sps: sample per symbol
    :return: 2-d array after inserting zeroes between symbols
    '''
    assert symbol_x.ndim == 1
    symbol_x.shape = -1, 1
    symbol_x = np.tile(symbol_x, (1, sps))
    symbol_x[:, 1:] = 0
    symbol_x.shape = 1, -1
    return symbol_x


def power_meter(signal: Signal.Signal, unit):
    if unit == 'dbm':
        return 10 * np.log10((np.sum(np.mean(np.abs(signal[:]) ** 2, axis=1))) * 1000)
    else:
        return np.sum(np.mean(np.abs(signal[:]) ** 2, axis=1))


def scatterplot(signal, pol_number):
    if pol_number == 2:
        plt.figure(figsize=(12, 6))
    if pol_number == 1:
        plt.figure(figsize=(6, 6))

    for i in range(pol_number):
        plt.subplot(1, pol_number, i + 1, aspect='equal')
        ibranch = signal[i, :].real
        qbranch = signal[i, :].imag
        plt.scatter(ibranch, qbranch, marker='o', color='b',s=1)
    plt.show()


def spectrum_analyzer(signal, fs=None):
    '''

    :param signal: signal object or ndarray
    :return: None
    '''

    if isinstance(signal, np.ndarray):
        assert fs is not None
        sample = signal
    else:
        fs = signal.fs_in_fiber

        sample = signal[:]

    pol_number = sample.shape[0]
    plt.figure(figsize=(20, 6))
    for i in range(pol_number):
        plt.subplot(1, pol_number, i + 1)
        [f, pxx] = welch(sample[i, :], fs, nfft=2048, detrend=False, return_onesided=False)
        plt.plot(f / 1e9, 10 * np.log10(np.abs(pxx)))
        plt.xlabel('Frequency [GHZ]')
        plt.ylabel('Power Spectrem Density [db/Hz]')
    plt.show()

def lamb2freq(lam):
    '''

    :param lam: wavelength [m]
    :return: frequence [Hz]
    '''
    return c / lam




def freq2lamb(freq):
    '''

    :param freq: frequence [Hz]
    :return: lambda:[m]
    '''
    return c / freq