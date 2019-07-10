import copy
from collections import Iterable

import numpy as np
from scipy.io import loadmat
from scipy.constants import c
from typing import List
from ..tool.tool import freq2lamb
import os
import Ocspy.qamdata

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Signal(object):
    '''
        This class is the base class of Signal,describe the base property of a signal in fiber
    '''

    def __init__(self, symbol_rate: float = 35e9, symbol_length: float = 65536,
                 mf: str = '16qam', sps_for_dsp: int = 2,
                 sps_in_fiber: int = 4, center_frequency: float = 193.1e12,
                 unit_rate: str = 'Hz', unit_freq: str = 'Hz', launch_power: float = 0, is_dp=True
                 ):

        if unit_rate.lower() == 'hz':
            unit_rate = 1
        elif unit_rate.lower() == 'ghz':
            unit_rate = 1e9
        else:
            raise Exception('The unit of baudrate must be Hz or GHz')

        if unit_freq.lower() == 'hz':
            unit_freq = 1
        elif unit_freq.lower() == 'thz':
            unit_freq = 1e12

        self.symbol_rate = symbol_rate * unit_rate
        self.symbol_length = symbol_length
        self.mf = mf
        self.sps_in_dsp = sps_for_dsp
        self.sps_in_fiber = sps_in_fiber
        self.center_frequency = center_frequency * unit_freq
        self.is_dp = True

        self.__ds_indsp = None
        self.ds_infiber = None
        self.launch_power = launch_power

    def __getitem__(self, index):

        assert self.ds_infiber is not None
        return self.ds_infiber[index]

    def __setitem__(self, index, value):
        if self.ds_infiber == None:
            self.ds_infiber[index] = np.at_least2d(value)
        else:
            self.ds_infiber[index] = value

    @property
    def fs(self):
        if self.sps_in_dsp is None:
            return 0
        return self.symbol_rate * self.sps_in_dsp

    @property
    def fs_in_fiber(self):
        return self.symbol_rate * self.sps_in_fiber

    @property
    def pol_number(self):
        return self.ds_infiber.shape[0]

    @property
    def shape(self):
        assert self.ds_infiber is not None
        return self.ds_infiber.shape

    def __str__(self):

        string = f'Baud Rate is   {self.symbol_rate / 1e9} [GHz]\t\n' \
            f'Modulation Format is {self.mf}\t\n' \
            f'Launch power is {self.launch_power}[dbm]\t\n' \
            f'center_frequency {self.center_frequency / 1e12} [THz]'
        #                  f'OSNR is {self.osnr}'
        return string

    def __repr__(self):
        return self.__str__()


class QamSignal(Signal):

    def __init__(self, symbol_rate: float = 35e9, symbol_length: float = 65536,
                 mf: str = '16-qam', sps_for_dsp: int = 2,
                 sps_in_fiber: int = 4, center_frequency: float = 193.1e12,
                 unit_rate: str = 'Hz', unit_freq: str = 'Hz', launch_power: float = 0, is_dp=True,
                 from_demux=False):

        super().__init__(symbol_rate=symbol_rate, symbol_length=symbol_length, mf=mf, sps_for_dsp=sps_for_dsp,
                         sps_in_fiber=sps_in_fiber,
                         center_frequency=center_frequency, unit_rate=unit_rate, unit_freq=unit_freq,
                         launch_power=launch_power, is_dp=is_dp)

        self.from_demux = from_demux
        if not self.from_demux:
            self.__init()

    def __init(self):
        if self.mf == 'qpsk':
            order = 4
        else:
            order = self.mf.split('-')[0]
            order = int(order)

        if self.is_dp:
            self.message = np.random.randint(low=0, high=order, size=(2, self.symbol_length))
        else:
            self.message = np.random.randint(low=0, high=order, size=(1, self.symbol_length))

        qam_data = (f'{base_path}/' + 'qamdata/' + str(order) + 'qam.mat')
        qam_data = loadmat(qam_data)['x']

        symbol = np.zeros(self.message.shape, dtype=np.complex)
        symbol = np.atleast_2d(symbol)
        for i in range(symbol.shape[0]):
            for msg in np.unique(self.message[i, :]):
                symbol[i, self.message[i, :] == msg] = qam_data[0, msg]

        self._symbol = symbol

    @property
    def symbol(self):
        return self._symbol



class SignalFromNumpyArray(Signal):

    def __init__(self, array_sample, symbol_rate, mf, symbol_length, sps_in_fiber, sps, **kwargs):
        '''

        :param array_sample: nd arrary
        :param symbol_rate: [Hz]
        :param mf: modulation format
        :param symbol_length: the length of transimitted symbol
        :param sps_in_fiber: the samples per symbol in fiber
        :param sps: the sampes per symbol in DSP

        This function will read samples from array_sample, the array_sample should be propogated in fiber

        The center frequence is the center frequence, and the absolute frequence is from left to right in wdm comb

        The relative frequence i.e base band equivalent is set a property
        '''
        super(SignalFromNumpyArray, self).__init__(
            **dict(symbol_rate=symbol_rate, mf=mf, symbol_length=symbol_length, sps_in_fiber=sps_in_fiber, sps=sps))
        self.data_sample_in_fiber = array_sample

        if 'tx_symbol' in kwargs:
            self.tx_symbol = kwargs['tx_symbol']
        if 'rx_symbol' in kwargs:
            self.rx_symbol = kwargs['rx_symbol']

        if 'center_frequence' in kwargs:
            self.center_frequence = kwargs['center_frequence']

        if 'aboslute_frequence' in kwargs:
            self.absolute_frequence = kwargs['absolute_frequence']

    @property
    def relative_frequence(self):
        if hasattr(self, 'center_frequence') and hasattr(self, 'absolute_frequence'):
            return self.absolute_frequence - self.center_frequence
        else:
            raise Exception(
                'The frequence not provided enough, please ensure the center_frequency and the absolute_frequenc are all provided')


class WdmSignal(object):
    '''
        WdmSignal Class
    '''

    def __init__(self, signals: List[Signal], grid_size, multiplexed_filed):
        '''

        :param signals: list of signal
        :param grid_size: [Hz],if conventional grid_size,this will be a number or a list of same value
        if the elestical wdm, the grid size of each channel is different.
        :param center_frequence: Hz

        The WdmSignal class should not be used outside usually, in mux function, it will be created automatically.

        '''
        self.signals = signals
        self.grid_size = grid_size
        self.center_frequence = (
                                        np.min(self.absolute_frequences) + np.max(self.absolute_frequences)) / 2
        self.center_wave_length = freq2lamb(self.center_frequence)
        self.__multiplexed_filed = np.atleast_2d(multiplexed_filed)

    def __setitem__(self, slice, value):

        assert self.__multiplexed_filed is not None
        self.__multiplexed_filed[slice] = value

    def __getitem__(self, slice):
        assert self.__multiplexed_filed is not None
        return self.__multiplexed_filed[slice]

    def __len__(self):
        return len(self.signals)

    # @property
    # def center_frequence(self):
    #     frequences = self.absolute_frequences()
    #     return (np.max(frequences)+np.min(frequences))/2

    @property
    def pol_number(self):
        return self.__multiplexed_filed.shape[0]

    @property
    def shape(self):
        return self.__multiplexed_filed.shape

    @property
    def fs_in_fiber(self):
        return self.signals[0].fs_in_fiber

    @property
    def sample_number_in_fiber(self):
        return self.__multiplexed_filed.shape[1]

    @property
    def lam(self):
        lambdas = []
        for signal in self.signals:
            lambdas.append(signal.center_wave_length)
        return lambdas

    @property
    def data_sample_in_fiber(self):
        return self.__multiplexed_filed

    @data_sample_in_fiber.setter
    def data_sample_in_fiber(self, value):
        self.__multiplexed_filed = value

    @property
    def mfs(self):
        mfs = []
        for sig in self.signals:
            mfs.append(sig.mf)
        return mfs

    @property
    def absolute_frequences(self):
        '''

        :return:
        '''
        return np.array([signal.center_frequence for signal in self.signals])

    @property
    def relative_frequences(self):
        '''

        :return: frequences centered in zero frequence , an ndarray
        '''
        return self.absolute_frequences - self.center_frequence

    def __str__(self):

        string = f"center_frequence is {self.center_frequence * 1e-12} [THZ]\t\n" \
                 f"power is {self.measure_power_in_fiber()} [dbm] \t\n" \
                 f"power is {self.measure_power_in_fiber(unit='w')} [w]\t\n"

        return string

    def measure_power_in_fiber(self, unit='dbm'):

        power = 0
        for i in range(self.pol_number):
            power += np.mean(np.abs(self[i, :]) ** 2)

        if unit == 'dbm':
            power = power * 1000
            power = 10 * np.log10(power)

        return power


class WdmSignalFromArray(object):

    def __init__(self, field, frequencys, fs_in_fiber, signal_under_study: List, signal_index):
        '''

        :param field:   WDM multiplexed fiedl
        :param center_freq: Hz will be caculated from frequencys
        :param frequencys: Hz each channel's frequencys, the passband frequence
        :param fs_in_fiber: Hz
        :param signal_index: the index of signal in original wdmsignal
        :param signal_under_study: the signal to study
        '''
        self.__filed = field
        self.fs_in_fiber = fs_in_fiber
        self.frequences = frequencys
        self.center_frequence = np.mean(
            [np.max(frequencys), np.min(frequencys)])
        self.center_wave_length = freq2lamb(self.center_frequence)  # m
        self.signal_under_study = signal_under_study
        self.signal_index = signal_index

    def __setitem__(self, slice, value):

        assert self.__filed is not None
        self.__filed[slice] = value

    def __getitem__(self, slice):

        assert self.__filed is not None
        return self.__filed[slice]

    @property
    def pol_number(self):
        return self.__filed.shape[0]

    def get_signal(self, signal_index):
        '''

        :param signal_index:signal index in original wdm signal
        :return: ith channel's signal
        '''
        if not isinstance(self.signal_index, Iterable):
            self.signal_index = [self.signal_index]

        index = self.signal_index.index(signal_index)
        return self.signal_under_study[index]

    @property
    def shape(self):
        return self.__filed.shape

    @property
    def sample_number_in_fiber(self):
        return self.__filed.shape[1]

    @property
    def lam(self):
        lambdas = []
        for freq in self.frequences:
            lambdas.append(freq2lamb(freq))
        return lambdas

    @property
    def data_sample_in_fiber(self):
        return self.__filed

    @data_sample_in_fiber.setter
    def data_sample_in_fiber(self, value):
        self.__filed = value

    @property
    def absolute_frequences(self):
        '''

        :return:
        '''
        return self.frequences

    @property
    def relative_frequences(self):
        '''

        :return: frequences centered in zero frequence , an ndarray
        '''
        return self.absolute_frequences - self.center_frequence

    def __str__(self):

        string = f"center_frequence is {self.center_frequence * 1e-12} [THZ]\t\n" \
                 f"power is {self.measure_power_in_fiber()} [dbm] \t\n" \
                 f"power is {self.measure_power_in_fiber(unit='w')} [w]\t\n"

        return string

    def measure_power_in_fiber(self, unit='dbm'):

        power = 0
        for i in range(self.pol_number):
            power += np.mean(np.abs(self[i, :]) ** 2)

        if unit == 'dbm':
            power = power * 1000  # convert to mw
            power = 10 * np.log10(power)

        return power

