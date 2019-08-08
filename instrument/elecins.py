from scipy.signal import fftconvolve

from ..Filter import rrcfilter
from ..signal_interface.signal import Signal
from resampy import resample

import numpy as np
from ..tools import upsample


class PulseShaping(object):
    '''
        Perform Pluse Shaping. need a dict to construct, the construct should contain:
            key:pulse_shaping
                span
                sps
                alpha

    '''

    def __init__(self, span, sps, alpha):
        self.span = span
        self.sps = sps
        self.alpha = alpha
        assert divmod(self.span * self.sps, 2)[1] == 0
        self.number_of_sample = self.span * self.sps
        self.delay = self.span / 2 * self.sps

        self.filter_tap = np.atleast_2d(rrcfilter(alpha, span, sps))

    def __call__(self, signal_interface: Signal):
        '''

        :param signal_interface: signal object to be pulse shaping,because a reference of signal object is passed
        so the filter is in place
        :return: None

        '''

        # print("---begin pulseshaping ---,the data sample will be set")
        # upsample by insert zeros

        signal_interface.ds = np.zeros(
            (signal_interface.symbol.shape[0], signal_interface.sps * signal_interface.symbol_length))
        signal_interface.ds = np.asarray(
            signal_interface.ds, dtype=np.complex)
        for i in range(signal_interface.symbol.shape[0]):
            signal_interface.ds[i, :] = upsample(signal_interface.symbol[i, :],
                                                 signal_interface.sps)[0, :]

        temp = []
        for i in range(signal_interface.ds.shape[0]):
            temp.append(
                fftconvolve(signal_interface.ds[i, :], self.filter_tap[0, :], mode='full'))

        # tempy = convolve(self.filter_tap[0, :], signal_interface.ds[1, :])
        # temp_signal = np.array([tempx, tempy])
        temp_signal = np.array(temp)
        # compensate group delay
        temp_signal = np.roll(temp_signal, -int(self.delay), axis=1)

        signal_interface.ds = temp_signal[:,
                              :signal_interface.sps * signal_interface.symbol_length]
        return signal_interface


class ADC(object):

    def __call__(self, signal, sps_in_fiber, sps):
        # from resampy import resample

        tempx = resample(signal[0], sps_in_fiber, sps, filter='kaiser_fast')
        tempy = resample(signal[1], sps_in_fiber, sps, filter='kaiser_fast')
        new_sample = np.array([tempx, tempy])

        return new_sample


class DAC(object):

    def __init__(self, is_quanti,clipping_ratio=None, resolution_bits=None ):
        self.clipping_ratio = clipping_ratio
        self.resolution_bits = resolution_bits
        self.is_quanti = is_quanti

    def __call__(self, signal: Signal):
        tempx = resample(signal.ds, signal.sps, signal.sps_in_fiber, axis=1, filter='kaiser_fast')
        signal[:] = tempx

        # signal[:] = self.quantization(signal[:])
        if self.is_quanti:
            assert self.clipping_ratio is not None
            assert self.resolution_bits is not None

            signal[0, :] = self.quantization(signal[0, :].real) + 1j * self.quantization(signal[0, :].imag)
            signal[1, :] = self.quantization(signal[1, :].real) + 1j * self.quantization(signal[1, :].imag)

        return signal

    def quantization(self, samples):
        samples = np.atleast_2d(samples)
        power = np.mean(samples.real ** 2 + samples.imag ** 2)
        A = 10 ** (self.clipping_ratio / 20) * np.sqrt(power)

        codebook = np.linspace(-A, A, 2 ** self.resolution_bits, endpoint=True)

        partition = codebook - (codebook[1] - codebook[0]) / 2
        partition = partition[1:]
        partition = np.atleast_2d(partition)
        codebook = np.atleast_2d(codebook)

        _, samples_quan = _quantize(samples, partition, codebook)
        return samples_quan


def _quantize(samples, partition, codebook):
    # samples = np.atleast_2d(samples)
    # codebook = np.atleast_2d(codebook)
    nrows, ncols = samples.shape
    indx = np.zeros((nrows, ncols), dtype=np.int64)

    for i in partition[0]:
        indx[samples > i] = indx[samples > i] + 1
        # indx = indx + np.array(samples > i,dtype=np.int64)
    quantv_signal = np.zeros_like(samples, dtype=np.float64)

    for i in range(quantv_signal.shape[0]):
        quantv_signal[i] = codebook[0, indx[i]]

    return indx, quantv_signal


class IqModulator(object):
    def __init__(self, vpi, insert_loss, extract, laser_power, vbais_i, vbais_q):
        self.vpi = vpi
        self.insert_loss = insert_loss
        self.extract = extract
        self.laser_power = laser_power
        self.vbais_i = vbais_i
        self.vbais_q = vbais_q

    def __call__(self, signal: Signal):
        ibranch = signal[:].real
        qbranch = signal[:].imag
        Er = 10 ** (self.extract / 10)
        gamma = (1 - 1 / np.sqrt(Er)) / 2
        for i in range(signal.is_pol + 1):
            ibranch[i] = self.laser_power / (10 ** (self.insert_loss / 20)) * (
                    gamma * np.exp(1j * np.pi * (ibranch[i] + self.vbais_i) / self.vpi) +
                    (1 - gamma) * np.exp(-1j * np.pi * (ibranch[i] + self.vbais_i) / self.vpi))
            qbranch[i] = self.laser_power / (10 ** (self.insert_loss / 20)) * (
                        gamma * np.exp(1j * np.pi * (qbranch[i] + self.vbais_q) / self.vpi) +
                        (1 - gamma) * np.exp(-1j * np.pi * (qbranch[i] + self.vbais_q) / self.vpi))

        samples = (ibranch + np.exp(1j * np.pi / 2) * qbranch) / 2
        signal[:] = samples
        return signal


if __name__ == '__main__':
    pass
