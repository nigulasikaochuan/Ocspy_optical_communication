import warnings

from scipy.constants import c
from scipy.interpolate import interp1d
from scipy.signal import welch, lfilter

from signal import signal
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

from .signal_interface import QamSignal, WdmSignal


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


def power_meter(signal: Union[QamSignal,WdmSignal], unit):
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
        plt.scatter(ibranch, qbranch, marker='o', color='b', s=1)
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


class OpticalSpectrumAnalyzer(object):

    def __init__(self, resolution_nm, noise_tor):
        self.resolution_nm = resolution_nm
        self.noise_tol = noise_tor

    def resolution_hz(self, center_wavelenght):
        '''

        :param center_wavelenght: the unit is m
        :return: the resolution of the OSA
        '''
        resolution = c / (center_wavelenght - self.resolution_nm * 1e-9 / 2) - c / (
                center_wavelenght + self.resolution_nm * 1e-9 / 2)

        return resolution

    def __call__(self, signal: Union[QamSignal, WdmSignal]):
        '''

        :param signal:
        :return: ydbm at y(hz) or ly in wavelength(m)
        '''
        power_spec = np.fft.fft(signal[:], axis=1)
        power_spec = power_spec.abs() ** 2
        power_spec = np.sum(power_spec, axis=0, keepdims=True)

        resolution_hz = self.resolution_hz(signal.center_wavelength)
        freq_vector = signal.freq_vector
        power_spec = power_spec / (signal.fs_in_fiber * power_spec.shape[1] / self.resolution_hz())
        df = np.diff(freq_vector)[0]
        window_length = np.round(resolution_hz / df)
        if divmod(window_length, 2)[1] == 0:
            window_length += 1
        pfilter = lfilter(np.ones(window_length) / window_length, 1, power_spec, axis=1)
        pfilter = np.roll(pfilter, 0 - (window_length - 1) / 2)

        center_frequency = signal.center_frequency
        fy = center_frequency + freq_vector
        function = interp1d(freq_vector, pfilter)
        Y = function(fy - signal.center_frequency)
        ly = c / fy
        ydbm = 10 * np.log10(Y / 1e-3)

        return ydbm, fy, ly

    def est_osnr(self, signal: Union[signal, QamSignal]):
        ydbm, fy, ly = self(signal)
        self.noise_tol = [self.noise_tol] if not isinstance(self.noise_tol, list) else self.noise_tol
        idx = np.argmax(ydbm)

        for tol in self.noise_tol:
            idx1 = np.abs(np.diff(ydbm[:idx]) < tol)
            idx2 = np.abs(np.diff(ydbm[idx:]) < tol)
            if np.any(idx1) or np.any(idx2):
                break
        else:
            warnings.warn('can not find noise floor')
            return
        idx1 = idx - idx1[0]
        idx2 = idx + idx2[0]

        noise = interp1d(ly[idx1], ydbm[idx1])
        noise2 = interp1d(ly[idx2], ydbm[idx2])
        noise = noise(ly[idx])
        noise2 = noise2(ly[idx])
        noise = np.hstack((noise, noise2))

        sn = 10 ** np.log10(np.max(ydbm) / 10)
        n = 10 ** (noise / 10)
        osnrdb = 10 * np.log10(sn / n - 1)
        osnrdb_125ghz = self.convert_osnr(osnrdb, signal.center_wavelength)
        return osnrdb_125ghz

    def convert_osnr(self, osnrdb, wavelength):
        osnrdb = osnrdb + 10 * np.log10(self.resolution_hz(wavelength) / 12.5e9)
        return osnrdb
