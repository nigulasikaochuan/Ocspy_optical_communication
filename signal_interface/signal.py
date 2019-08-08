
import copy
import os
import warnings
import numpy as np

from typing import List, Union

from numpy.fft import fftfreq
from scipy.constants import c

GHz = 1e9
Hz = 1
Km = 1000
m = 1
Thz = 1e12
unit_dict = dict(ghz=GHz, hz=Hz, km=Km, m=m, thz=Thz)


class Signal(object):
    def __init__(self, center_frequency, unit_freq, baudrate: float ,unit: str , sps: int = 2, sps_in_fiber: int = 4,
                 mf: str = '16qam'

                 , is_pol: bool = True,  launch_power: float = 0,

                 symbol_length=2 ** 16):
        warnings.warn(
            'Please ensure the unit is correct, the default is Ghz for baud rate and Thz for center frequency')
        self.baudrate = baudrate * unit_dict[unit.lower()]
        self.sps = sps
        self.sps_in_fiber = sps_in_fiber
        self.mf = mf
        self.is_pol = is_pol
        self.unit = unit
        self.unit_freq = unit_freq
        self.launch_power = launch_power
        self.__center_frequency = center_frequency * unit_dict[unit_freq.lower()]
        self.symbol_length = symbol_length
        self.ds = None
        self.ds_in_fiber = None
        self.ase_power = 0

    def __str__(self):
        string = f'baudrate : {self.baudrate/unit_dict[self.unit]} {self.unit}\t\n' \
            f'signal power is {self.launch_power} dbm\t\n' \
            f'signal power is {10 ** (self.launch_power / 10) / 1000} w\t\n' \
            f'center_frequency is {self.center_frequency/unit_dict[self.unit_freq]} {self.unit_freq}\t\n' \
            f'center wavelength is {c / self.center_frequency} m\t\n' \
            f'modulation format is {self.mf}'
        return string

    def __repr__(self):
        return self.__str__()

    @property
    def center_frequency(self):
        return self.__center_frequency

    @property
    def center_wavelength(self):
        return c / self.center_frequency

    def __setitem__(self, index, value):
        # assert self.ds_in_fiber is not None
        if self.ds_in_fiber is None:
            self.ds_in_fiber = np.atleast_2d(value)
        else:

            self.ds_in_fiber[index] = value

    def __getitem__(self, index):
        assert self.ds_in_fiber is not None
        return np.atleast_2d(self.ds_in_fiber[index])

    @property
    def shape(self):
        return self.ds_in_fiber.shape

    @property
    def symbol(self):
        return None

    @property
    def fs_in_fiber(self):
        return self.baudrate * self.sps_in_fiber


    @property
    def freq_vector(self):
        assert self.ds_in_fiber is not None
        freq_vector = np.atleast_2d(fftfreq(self.shape[1], 1 / self.fs_in_fiber))
        return freq_vector

class QamSignal(Signal):

    def __init__(self, is_init: bool = True, **kwargs, ):
        super(QamSignal, self).__init__(**kwargs)
        self.__msg = None
        self._symbol = np.zeros((self.is_pol + 1, self.symbol_length), dtype=np.complex)
        self.constl = None

        if is_init:
            self.__init()

    def __init(self):
        if self.mf == '16qam':
            self.order = 16
        elif self.mf == 'qpsk':
            self.order = 4
        elif self.mf == '8qam':
            self.order = 8
        elif self.mf == '32qam':
            self.order = 32
        elif self.mf == '64qam':
            self.order = 64
        self.__msg = np.random.randint(0, high=self.order, size=self._symbol.shape)

        self.__map()
        self.__msg = np.atleast_2d(self.__msg)
        self._symbol = np.atleast_2d(self._symbol)

    def __map(self):
        BASE_DIR = os.path.dirname(__file__)
        constl = np.load(f'{BASE_DIR}/{self.order}qam.npy')
        for i in range(self._symbol.shape[0]):
            for msg in range(0, self.order):
                self._symbol[i, self.__msg[i] == msg] = constl[0, msg]

        self.constl = constl

    @property
    def symbol(self):
        return self._symbol

    @property
    def msg(self):
        return self.__msg


class WdmSignal(object):
    def __init__(self, signals: List[Signal], samples):

        if not isinstance(signals, list):
            signals = [signals]

        fs_in_fiber = []
        for signal in signals:
            fs_in_fiber.append(signal.fs_in_fiber)
        if np.any(np.diff(fs_in_fiber)):
            raise Exception("The fs of all signal should be the same")

        self.fs_in_fiber = signals[0].fs_in_fiber

        self.symbol_rates = [signal.baudrate for signal in signals]
        self.mfs = [signal.mf for signal in signals]
        self.frequencies = [signal.center_frequency for signal in signals]
        if len(np.unique(self.frequencies)) == 1: \
                warnings.warn('please assure the wdm channel is only 1')

        self.__field = samples
        self.ase_power = 0

    @property
    def abs_freq(self):
        return self.frequencies

    @property
    def rela_freq(self):
        center_freq = max(self.frequencies) + min(self.frequencies)
        center_freq /= 2
        return [freq - center_freq for freq in self.frequencies]

    @property
    def center_wavelength(self):
        return c / ((max(self.frequencies) + min(self.frequencies)) / 2)

    @property
    def shape(self):
        return self[:].shape

    @property
    def center_freq(self):
        return (max(self.frequencies) + min(self.frequencies)) / 2

    def __setitem__(self, key, value):
        assert self.__field is not None
        self.__field[key] = value

    def __getitem__(self, key):
        assert self.__field is not None
        return self.__field[key]

    @property
    def is_pol(self):
        return self[:].shape[0] == 2
    @property
    def freq_vector(self):
        assert self.__field is not None
        freq_vector = np.atleast_2d(fftfreq(self.shape[1], 1 / self.fs_in_fiber))
        return freq_vector

if __name__ == '__main__':
    pass
