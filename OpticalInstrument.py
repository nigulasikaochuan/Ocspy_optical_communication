import warnings
from typing import List
import numba
import tqdm
from numpy.fft import fftfreq
from scipy.constants import h, c
from scipy.fftpack import ifft, fft
from scipy.special import erf

from Signal import Signal, WdmSignal
import numpy as np


def mux_signal(signals: List[Signal]):
    '''

        :param signals: signals that will be
        :param grid_size: the grid size of the wdm signal
        :return:
        '''
    fs_in_fiber = []
    for signal in signals:
        fs_in_fiber.append(signal.fs_in_fiber)

    if np.any(np.diff(np.array(fs_in_fiber))):
        assert "the sampling frequence of each signal in fiber should be the same"

    absolute_frequences = []
    for signal in signals:
        absolute_frequences.append(signal.center_frequency)

    absolute_frequences = np.array(absolute_frequences)
    freq = np.min(absolute_frequences) + np.max(absolute_frequences)
    freq = freq / 2

    relative_frequence = np.array(absolute_frequences) - freq
    # 从左到右开始复用
    # 每个wdm信道的采样频率必须保持一致
    # 每个wdm信道的样本点个数也要保持一致,截掉长的采样序列的尾部

    sample_length = []
    for signal in signals:
        sample_length.append(signal.shape[1])

    if np.any(np.diff(sample_length)):
        warnings.warn("the number of sample in fiber are not the same, the longer will be "
                      "trancted")
        min_length = min(sample_length)

        for signal in signals:
            number_to_del = signal.shape[1] - min_length
            if number_to_del == 0:
                continue
            else:
                signal.data_sample_in_fiber = signal.data_sample_in_fiber[:, 0: - number_to_del]

    sample_number = signals[0].shape[1]
    fs = signals[0].fs_in_fiber
    t_array = np.arange(0, sample_number) * (1 / fs)

    # pol_number = signals[0].pol_number
    channel_number = len(signals)
    # sample_length = signals[0][:].shape[1]

    # samples = np.zeros((pol_number, channel_number, sample_length), dtype=np.complex128)
    wdm_data_sample = 0 + 0j

    for ch_index in tqdm.tqdm(range(channel_number), ascii=True):
        wdm_data_sample = wdm_data_sample + _mux_signal2(signals[ch_index][:], t_array, relative_frequence[ch_index])

        # wdm_data_sample = mux_signal(samples, t_array, relative_frequence.reshape(-1, 1))

    wdm_signal = WdmSignal(signals, wdm_data_sample)

    return wdm_signal


@numba.njit("complex128[:,:](complex128[:,:],float64[:],float64)", cache=True)
def _mux_signal2(samples, t_array, relative_frequence):
    exp_factor = np.exp(1j * 2 * np.pi * relative_frequence * t_array)
    samples = samples * exp_factor
    return samples


def demux_signal(wdm_signal: WdmSignal, signal_index=None):
    '''

    This static function is used to demultiplex a wdm_signal, the signal_index of wdm_signal will be obtained, it
    will be shift to center_frequency, an ideal low pass filter will be used to remove other signal that is not
    interested in. Note CD should do first

    :param wdm_signal: the wdm signal to be demulitplexed
    :param signal_index: which signal want to get,return signal objectl,if None.all channel will be returned
    :return: a QamSignal
    '''
    assert isinstance(wdm_signal, WdmSignal)

    symbol_rate = wdm_signal.symbol_rates[signal_index]

    frequences = wdm_signal.rela_freq
    tarray = np.arange(0, wdm_signal[:].shape[1]) * (1 / wdm_signal.fs_in_fiber)

    # temp = wdm_signal[:]
    temp = np.zeros_like(wdm_signal[:])

    for i in range(temp.shape[0]):
        temp[i, :] = wdm_signal[i, :] * np.exp(-1j * 2 * np.pi * frequences[signal_index] * tarray)

    # center_frequence = wdm_signal.relative_frequence[signal_index]

    # LowPassFilter.inplace_ideal_lowpass(temp,pos_freq,neg_freq,fs_in_fiber=wdm_signal.fs_in_fiber)
    signal = temp
    signal = np.atleast_2d(signal)
    return signal


class Edfa:

    def __init__(self, gain_db, nf, is_ase=True, mode='ConstantGain', expected_power=0):
        '''

        :param gain_db:
        :param nf:
        :param is_ase: 是否添加ase噪声
        :param mode: ConstantGain or ConstantPower
        :param expected_power: 当mode为ConstantPoower  时候，此参数有效
        '''

        self.gain_db = gain_db
        self.nf = nf
        self.is_ase = is_ase
        self.mode = mode
        self.expected_power = expected_power

    def one_ase(self, signal:Signal, gain_lin=None):
        '''

        :param signal:
        :return:
        '''
        lamb = signal.center_wavelength
        if gain_lin is None:
            one_ase = (h * c / lamb) * (self.gain_lin * self.nf_lin - 1) / 2
        else:
            one_ase = (h * c / lamb) * (gain_lin * self.nf_lin - 1) / 2
        return one_ase

    @property
    def gain_lin(self):
        return 10 ** (self.gain_db / 10)

    @property
    def nf_lin(self):
        return 10 ** (self.nf / 10)

    def traverse(self, signal:Signal):
        if self.mode == 'ConstantPower':
            #             raise NotImplementedError("Not implemented")
            signal_power = np.mean(np.abs(signal[0, :]) ** 2) + np.mean(
                np.abs(signal[1, :]) ** 2)
#             print(signal_power)
            desired_power_linear = (10 ** (self.expected_power / 10)) / 1000
            linear_gain = desired_power_linear / signal_power
            self.gain_db = 10*np.log10(linear_gain)
            signal[:] = np.sqrt(linear_gain) * signal[:]

        if self.mode == 'ConstantGain':
            signal[:] = np.sqrt(self.gain_lin) * signal[:]


        noise = self.one_ase(signal) * signal.fs_in_fiber
        each_pol_power = noise
        if self.is_ase:
            noise_sample = np.random.randn(*(signal[:].shape)) + 1j * np.random.randn(*(signal[:].shape))

            noise_sample = np.sqrt(each_pol_power / 2) * noise_sample
            signal[:] = signal[:]+noise_sample
        signal.ase_power += each_pol_power*2
        return signal

    def __call__(self, signal):
        self.traverse(signal)
        return signal

    def __str__(self):

        string = f"Model is {self.mode}\n" \
            f"Gain is {self.gain_db} db\n" \
            f"ase is {self.is_ase}\n" \
            f"noise figure is {self.nf}"
        return string

    def __repr__(self):
        return self.__str__()

class WSS(object):
    unit_dict = {'ghz':1,'hz':1e9}

    def __init__(self, frequency_offset, bandwidth, oft,unit):

        '''

        :param frequency_offset: value away from center [GHz]
        :param bandwidth: 3-db Bandwidth [Ghz]
        :param oft:GHZ
        '''
        self.frequency_offset = frequency_offset/WSS.unit_dict[unit.lower()]
        self.bandwidth = bandwidth/WSS.unit_dict[unit.lower()]
        self.oft = oft/WSS.unit_dict[unit.lower()]
        self.H = None
        self.freq = None

    def traverse(self, signal):

        sample = np.zeros_like(signal[:])
        for i in range(sample.shape[0]):
            sample[i, :] = signal[i, :]

        freq = fftfreq(len(sample[0, :]), 1 / signal.fs_in_fiber)
        freq = freq / 1e9
        self.freq = freq
        self.__get_transfer_function(freq)

        for i in range(sample.shape[0]):
            sample[i, :] = ifft(fft(sample[i, :]) * self.H)

        return sample

    def __call__(self, signal):
        sample = self.traverse(signal)
        signal[:] = sample
        return signal

    def __get_transfer_function(self, freq_vector):
        delta = self.oft / 2 / np.sqrt(2 * np.log(2))

        H = 0.5 * delta * np.sqrt(2 * np.pi) * (
                erf((self.bandwidth / 2 - (freq_vector - self.frequency_offset)) / np.sqrt(2) / delta) - erf(
            (-self.bandwidth / 2 - (freq_vector - self.frequency_offset)) / np.sqrt(2) / delta))

        H = H / np.max(H)

        self.H = H

    def plot_transfer_function(self, freq=None):
        import matplotlib.pyplot as plt
        if self.H is None:
            self.__get_transfer_function(freq)
            self.freq = freq
        index = self.H > 0.001
        plt.figure(figsize=(20, 6))
        plt.subplot(121)
        plt.scatter(self.freq[index], np.abs(self.H[index]), color='b', marker='o')
        plt.xlabel('GHz')
        plt.ylabel('Amplitude')
        plt.title("without log")
        plt.subplot(122)
        plt.scatter(self.freq[index], 10 * np.log10(np.abs(self.H[index])), color='b', marker='o')
        plt.xlabel('GHz')
        plt.ylabel('Amplitude')
        plt.title("with log")
        plt.show()

    def __str__(self):

        string = f'the center_frequency is {0 + self.frequency_offset}[GHZ] \t\n' \
            f'the 3-db bandwidth is {self.bandwidth}[GHz]\t\n' \
            f'the otf is {self.oft} [GHz] \t\n'
        return string

    def __repr__(self):
        return self.__str__()
