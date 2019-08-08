from functools import partial

import numba

from instrument.elecins import ADC
from Filter import ideal_lowpass
from instrument.optins import demux_signal
from signal_interface.signal import Signal
from .dsp import dual_pol_time_domain_lms_equalizer, superscalar, cd_compensation, MatchedFilter, syncsignal, \
    demap_to_msg_v2
from .dsp import cal_symbols_qam, cal_scaling_factor_qam, downsample, normal_sample
import numpy as np
from collections import namedtuple


class LMS(object):

    def __init__(self, kind='time', ntaps=21, learing_rate=0.001, niter=3, is_train=True, order=16):
        if kind == 'freq':
            raise NotImplementedError("only time domain supported now")

        self.hxx = None
        self.hyy = None
        self.hyx = None
        self.hxy = None
        self.errors = None
        self.ntaps = ntaps
        self.learning_rate = learing_rate
        self.niter = niter
        self.is_train = is_train
        self.constl = cal_symbols_qam(order) / np.sqrt(cal_scaling_factor_qam(order))[np.newaxis, :]

    def equalize(self, signal: Signal):
        trian_symbol = signal.symbol if self.is_train else None

        xsymbols, ysymbols, weight, errorsx, errorsy = dual_pol_time_domain_lms_equalizer(signal[:], self.ntaps,
                                                                                          signal.sps_in_fiber,
                                                                                          self.constl, trian_symbol,
                                                                                          self.learning_rate,
                                                                                          self.niter,
                                                                                          signal.symbol_length)

        self.hxx = weight[0][0][np.newaxis, :]
        self.hxy = weight[1][0][np.newaxis, :]
        self.hyx = weight[2][0][np.newaxis, :]
        self.hyy = weight[3][0][np.newaxis, :]

        equalized_symbol = np.vstack((xsymbols, ysymbols))
        self.errors_xpol = errorsx
        self.errors_ypol = errorsy

        return equalized_symbol

    def __call__(self, signal):
        equalized_symbol = self.equalize(signal)
        return equalized_symbol


# def superscalar(symbol_in, training_symbol, block_length, pilot_number, constl, g,filter_n=20):

class SuperScalar(object):

    def __init__(self, block_length, pilot_number, constl, g, filter_n, trian_symbol):
        self.block_length = block_length
        self.pilot_number = pilot_number
        self.constl = constl
        self.g = g
        self.filter_n = filter_n
        self.train_symbol = trian_symbol

    def __call__(self, symbols):
        Res = namedtuple('SuperScalr res', ['cpe_symbol', 'original_symbol'])

        x, y = superscalar(symbols, self.train_symbol, self.block_length,
                           self.pilot_number, self.constl, self.g, self.filter_n)
        res = Res(x, y)
        return res


class DspProcess(object):

    def __init__(self, is_lms=True, is_cpe=True, is_wdm=True, **config):
        '''

        :param is_lms:
        :param is_cpe:
        :param is_wdm:
        :param config: cofiguration dictionary
            matched_filter: roll_off
            signal:order int
            cd:spans
            lms:ntaps,learning_rate，niter,is_train

            cpe:block_length pilot_number constl g filter_n train_symbol
        '''
        self.is_wdm = is_wdm
        self.config = config
        self.roll_off = config['matched_filter']['roll_off']
        self.order = config['signal']['order']
        if config['cd']['spans'] is not None:
            self.cd = partial(cd_compensation, spans=config['cd']['spans'], inplace=False)

        if is_lms:
            ntaps = config['lms']['ntaps']
            learning_rate = config['lms']['learning_rate']
            niter = config['lms']['niter']
            is_train = config['lms']['is_train']
            order = config['lms']['order']
            self.lms = LMS('time', ntaps, learning_rate, niter, is_train, order)

        if is_cpe:
            block_length = config['cpe']['block_length']
            pilot_number = config['cpe']['pilot_number']
            constl = config['cpe']['constl']
            g = config['cpe']['g']
            filter_n = config['cpe']['filter_n']
            train_symbol = config['cpe']['train_symbol']
            self.superscalar = SuperScalar(block_length, pilot_number, constl, g, filter_n, train_symbol)
        self.noise_power = []
        self.snr = []
        self.error_rate_xpol = None
        self.error_rate_ypol = None

    def __call__(self, signal: Signal, signal_index=None, calc_ber=False):
        signal = self.cd(signal)
        if self.is_wdm:
            assert signal_index is not None
            signal_samples = demux_signal(signal, signal_index)
            symbol_rate = signal.symbol_rates[signal_index]
        else:
            signal_samples = signal[:]
            symbol_rate = signal.baudrate

        pos_freq = symbol_rate / 2 * (1 + self.roll_off)
        neg_freq = -pos_freq
        samples = ideal_lowpass(signal_samples, pos_freq, neg_freq, signal.fs_in_fiber)

        samples = ADC()(samples, signal.sps_in_fiber, signal.sps)
        samples = MatchedFilter(roll_off=self.roll_off, sps=2, span=1024)(samples)
        samples = normal_sample(samples)

        if hasattr(self, 'lms'):
            samples.sps_in_fiber = 2
            samples.symbol_length = signal.symbol_length
            samples.symbol = signal.symbol
            symbol_equalized = self.lms(samples)
            symbol_equalized = syncsignal(sample_rx=symbol_equalized, symbol_tx=signal.symbol, sps=1)

        else:
            symbol_equalized = downsample(samples, 2)

        if hasattr(self, 'superscalar'):
            symbol_final = self.superscalar(symbol_equalized)
        else:
            angle = np.angle(np.mean(symbol_equalized / signal.symbol, axis=1))
            symbol_final = symbol_equalized * np.exp(-1j * angle)

        symbol_final = normal_sample(symbol_final)
        noise = symbol_final - signal.symbol
        self.noise_power.append(np.mean(noise[0].real ** 2 + noise[0].imag ** 2))
        self.noise_power.append(np.mean(noise[1].real ** 2 + noise[1].imag ** 2))

        self.noise_power.append(np.sum(self.noise_power))

        self.snr = [10 * np.log10(1 / self.noise_power[0]), 10 * np.log10(1 / self.noise_power[1]),
                    10 * np.log10(2 / (self.noise_power[0] + self.noise_power[1]))]

        if calc_ber:
            self.msg = demap_to_msg_v2(symbol_final, self.order, False)
            self.error_rate_xpol = biterror_exp(self.order, self.msg[0], signal.msg[0])

            self.error_rate_ypol = biterror_exp(self.order, self.msg[1], signal.msg[1])


@numba.njit('int32[:](int32,int32)', cache=True)
def msg2bin(msg, number):
    x = np.ones(number, dtype=np.int32)

    for i in range(number):
        if i == 0:
            x[i] = divmod(msg, 2)[1]
            next_number = divmod(msg, 2)[0]
        else:
            x[i] = divmod(next_number, 2)[1]
            next_number = divmod(next_number, 2)[0]
    return x[::-1]


@numba.njit('float64(int32,int32[:],int32[:])', cache=True)
def biterror_exp(order, receive_msg, tx_msg):
    assert receive_msg.ndim == 1
    assert tx_msg.ndim == 1
    selected_msg_recv = receive_msg[receive_msg != tx_msg]
    selected_msg_tx = tx_msg[receive_msg != tx_msg]
    bitnumber = np.log2(order)
    assert selected_msg_recv.shape == selected_msg_tx.shape
    biterror_number = 0
    for cnt in range(len(selected_msg_tx)):
        recv_bin = msg2bin(selected_msg_recv[cnt], bitnumber)
        tx_bin = msg2bin(selected_msg_tx[cnt], bitnumber)
        biterror_number += np.sum(tx_bin != recv_bin)

    return biterror_number / bitnumber / len(receive_msg)
