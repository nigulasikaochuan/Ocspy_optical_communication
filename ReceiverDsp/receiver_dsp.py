import numpy as np
import numba
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import fftconvolve
from scipy.signal import correlate
from .dsp_tools import segment_axis, decision, cal_symbols_square_qam, cal_scaling_factor_qam, get_power, \
    cal_symbols_qam
from ..Base.SignalInterface import Signal
from ..Instrument.ElectricInstrument import PulseShaping
import os


class MatchedFilter(object):

    def __init__(self, roll_off=0.02, span=1024, sps=4):
        self.h = PulseShaping.rcosdesign(int(span * sps), roll_off, 1, int(sps))
        self.delay = int(span / 2 * sps)

    def match_filter(self, signal):
        x = signal.data_sample_in_fiber[0, :]
        y = signal.data_sample_in_fiber[1, :]
        x = fftconvolve(x, self.h)
        y = fftconvolve(y, self.h)

        x = np.roll(x, -self.delay)
        y = np.roll(y, -self.delay)
        x = x[:signal.sample_number_in_fiber]
        y = y[:signal.sample_number_in_fiber]
        return x, y

    def inplace_match_filter(self, signal):
        x, y = self.match_filter(signal)
        signal.data_sample_in_fiber = np.array([x, y])
        return signal

    def __call__(self, signal):
        self.inplace_match_filter(signal)

        return signal


def syncsignal(symbol_tx, sample_rx, sps):
    '''

        :param symbol_tx: 发送符号
        :param sample_rx: 接收符号，会相对于发送符号而言存在滞后
        :param sps: samples per symbol
        :return: 收端符号移位之后的结果

        # 不会改变原信号

    '''
    assert sample_rx.ndim == 1
    assert symbol_tx.ndim == 1
    assert len(sample_rx) >= len(symbol_tx)
    symbol_tx = np.atleast_2d(symbol_tx)[0, :]
    sample_rx = np.atleast_2d(sample_rx)[0, :]

    res = correlate(sample_rx[::sps], symbol_tx)

    index = np.argmax(np.abs(res))

    out = np.roll(sample_rx, sps * (-index - 1 + symbol_tx.shape[0]))
    return out


def cd_compensation(signal: Signal, spans, inplace=False):
    '''

    This function is used for chromatic dispersion compensation in frequency domain.
    The signal is Signal object, and a new sample is created from property data_sample_in_fiber

    :param signal: Signal object
    :param spans: Span object, the signal's has propagated through these spans
    :param inplace: if True, the compensated sample will replace the original sample in signal,or new ndarray will be r
    eturned

    :return: if inplace is True, the signal object will be returned; if false the ndarray will be returned
    '''
    center_wavelength = signal.center_wave_length
    freq_vector = fftfreq(signal[0, :].shape[0], 1 / signal.fs_in_fiber)
    omeg_vector = 2 * np.pi * freq_vector

    sample = np.zeros_like(signal[:])

    for i in range(signal.shape[0]):
        sample[i, :] = signal[i, :]

    if not isinstance(spans, list):
        spans = [spans]

    for span in spans:
        beta2 = -span.beta2(center_wavelength)
        dispersion = (-1j / 2) * beta2 * omeg_vector ** 2 * span.length
        for pol_index in range(sample.shape[0]):
            sample[pol_index, :] = ifft(fft(sample[pol_index, :]) * np.exp(dispersion))

    if inplace:
        signal[:] = sample
        return signal
    else:
        return sample


def super_scalar():
    '''
        implementing superscalar algorithm to recover signal's phase
    :return:
    '''
    pass


def __pll():
    '''
        using in super scalar, should not be called outside this file
    :return:
    '''
    pass


def __ml():
    '''
        using in super scalar, should not be called outside this file
    :return:
    '''
    pass


def over_lap_save(signal, h):
    '''

    :param signal: 1d-array
    :param h: filter's taps
    implement linear convolution using over-lap-save method
    :return:
    '''
    pass


def dual_frequency_lms_equalizer_block(signal, ntaps, sps, constl=None, train_symbol=None, mu=0.001, niter=3):
    pass


@numba.jit(cache=True)
def dual_pol_time_domain_lms_equalizer_pll(signal, ntaps, sps, constl=None, train_symbol=None, mu=0.001,
                                           niter=3, g=0.01, symbol_length=65536):
    weight = __dual_pol_init_lms_weight(ntaps)  # xx xy yx yy
    data_sample_in_fiber = signal[:]
    samplex = segment_axis(data_sample_in_fiber[0, :], ntaps, ntaps - sps)
    sampley = segment_axis(data_sample_in_fiber[1, :], ntaps, ntaps - sps)
    if train_symbol is not None:
        xsymbol = train_symbol[0, ntaps // 2 // sps:]
        ysymbol = train_symbol[1, ntaps // 2 // sps:]
        # xsymbol = np.roll(xsymbol,)

    xsymbols = np.zeros((1, symbol_length), dtype=np.complex128)
    ysymbols = np.zeros((1, symbol_length), dtype=np.complex128)
    errorsx = np.zeros((1, niter * samplex.shape[0]), dtype=np.float64)
    errorsy = np.zeros((1, niter * samplex.shape[0]), dtype=np.float64)
    pll_phase = np.zeros((2, samplex.shape[0]), dtype=np.complex128)
    pll_error = np.zeros((2, samplex.shape[0]), dtype=np.complex128)

    hxx = np.zeros((1, samplex.shape[0]),dtype = np.float32)
    hxy = np.zeros((1, samplex.shape[0]),dtype = np.float32)
    hyx = np.zeros((1, samplex.shape[0]),dtype = np.float32)
    hyy = np.zeros((1, samplex.shape[0]),dtype = np.float32)

    for j in range(niter):
        for i in range(samplex.shape[0]):
            xout = np.sum(samplex[i, ::-1] * weight[0][0]) + np.sum(sampley[i, ::-1] * weight[1][0])
            yout = np.sum(samplex[i, ::-1] * weight[2][0]) + np.sum(sampley[i, ::-1] * weight[3][0])
            xout_cpr = xout * np.exp(-1j * pll_phase[0, i])
            yout_cpr = yout * np.exp(-1j * pll_phase[1, i])

            pll_error[0, i] = (xout * np.conj(xout_cpr)).imag
            pll_error[1, i] = (yout * np.conj(yout_cpr)).imag

            pll_phase[0, i] = g * pll_error[0, i] + pll_phase[0, i]

            pll_phase[1, i] = g * pll_error[1, i] + pll_phase[1, i]
            #  is not None and j == 0:
            if train_symbol is not None:
                errorx = __calc_error_train(xout_cpr, symbol=xsymbol[i])
                errory = __calc_error_train(yout_cpr, symbol=ysymbol[i])
            else:
                assert constl is not None
                # assert constl.ndim ==1

                if constl.ndim != 2:
                    raise Exception("constl must be 2d")

                errorx = __calc_error_dd(xout_cpr, constl)
                errory = __calc_error_dd(yout_cpr, constl)

            weight[0][0] = weight[0][0] + mu * errorx * np.conj(samplex[i, ::-1])
            weight[1][0] = weight[1][0] + mu * errorx * np.conj(sampley[i, ::-1])

            weight[2][0] = weight[2][0] + mu * errory * np.conj(samplex[i, ::-1])
            weight[3][0] = weight[3][0] + mu * errory * np.conj(sampley[i, ::-1])

            if j == niter - 1:
                xsymbols[0, i] = xout_cpr
                ysymbols[0, i] = yout_cpr
                hxx[0,i] = weight[0][0,ntaps//2+1].real
                hxy[0,i] = weight[1][0,ntaps//2+1].real

                hyx[0,i] = weight[2][0,ntaps//2+1].real

                hyy[0, i] = weight[3][0, ntaps//2+1].real

            errorsx[0, i] = np.abs(errorx)
            errorsy[0, i] = np.abs(errory)

    return xsymbols[0, :], ysymbols[0, :], weight, errorsx, errorsy,(hxx,hxy,hyx,hyy)


@numba.jit(cache=True)
def dual_pol_time_domain_lms_equalizer(signal, ntaps, sps, constl=None, train_symbol=None, mu=0.001,
                                       niter=3, symbol_length=65536):
    '''

    :param signal:         signal to be equalized, a numpy array, 2d-array,each row represent a polarizaiton
    :param ntaps:          the tap of the filter
    :param sps:            sample per symbol
    :param constl:         the constellation, used in dd
    :param train_symbol:   symbol to train
    :param mu:             the learning rate, default to 0.001
    :param niter:          the number of equlizeing operation
    :return: equalized symbols, errors
    '''
    data_sample_in_fiber = signal[:]
    samplex = segment_axis(data_sample_in_fiber[0, :], ntaps, ntaps - sps)
    sampley = segment_axis(data_sample_in_fiber[1, :], ntaps, ntaps - sps)

    if train_symbol is not None:
        xsymbol = train_symbol[0, ntaps // 2 // sps:]
        ysymbol = train_symbol[1, ntaps // 2 // sps:]
        # xsymbol = np.roll(xsymbol,)

    weight = __dual_pol_init_lms_weight(ntaps)  # xx xy yx yy
    xsymbols = np.zeros((1, symbol_length), dtype=np.complex128)
    ysymbols = np.zeros((1, symbol_length), dtype=np.complex128)
    errorsx = np.zeros((1, niter * samplex.shape[0]), dtype=np.float64)
    errorsy = np.zeros((1, niter * samplex.shape[0]), dtype=np.float64)

    hxx = np.zeros((1,samplex.shape[0]))
    hxy = np.zeros((1, samplex.shape[0]))
    hyx = np.zeros((1, samplex.shape[0]))
    hyy = np.zeros((1, samplex.shape[0]))

    for j in range(niter):
        for i in range(samplex.shape[0]):
            xout = np.sum(samplex[i, ::-1] * weight[0][0]) + np.sum(sampley[i, ::-1] * weight[1][0])
            yout = np.sum(samplex[i, ::-1] * weight[2][0]) + np.sum(sampley[i, ::-1] * weight[3][0])

            if train_symbol is not None and j == 0:
                errorx = __calc_error_train(xout, symbol=xsymbol[i])
                errory = __calc_error_train(yout, symbol=ysymbol[i])
            else:
                assert constl is not None
                # assert constl.ndim ==1

                if constl.ndim != 2:
                    raise Exception("constl must be 2d")

                errorx = __calc_error_dd(xout, constl)
                errory = __calc_error_dd(yout, constl)

            weight[0][0] = weight[0][0] + mu * errorx * np.conj(samplex[i, ::-1])
            weight[1][0] = weight[1][0] + mu * errorx * np.conj(sampley[i, ::-1])

            weight[2][0] = weight[2][0] + mu * errory * np.conj(samplex[i, ::-1])
            weight[3][0] = weight[3][0] + mu * errory * np.conj(sampley[i, ::-1])

            if j == niter - 1:
                xsymbols[0, i] = xout
                ysymbols[0, i] = yout
            errorsx[0, i] = np.abs(errorx)
            errorsy[0, i] = np.abs(errory)

    return xsymbols, ysymbols, weight, errorsx, errorsy


@numba.jit(cache=True)
def __calc_error_dd(xout, constl):
    '''
        only use in dual_pol_time_domain_lms_equalizer, should not be used outside this file
        xout: a symbol to be decision,one element
        constl: must be 2d array
    '''
    symbol = decision(xout, constl)
    return symbol - xout


@numba.jit(cache=True)
def __calc_error_train(xout, symbol):
    '''
        only use in dual_pol_time_domain_lms_equalizer, should not be used outside this file
        xout: symbol in ,one element
        symbol: True symbol,one element
    '''
    error = symbol - xout
    return error


@numba.jit(cache=True)
def __dual_pol_init_lms_weight(ntaps):
    '''
     only use in dual_pol_time_domain_lms_equalizer, should not be used outside this file
     implement filter taps initialization

    :param ntaps:
    :return:
    '''
    # hxx = np.zeros((1, ntaps), dtype=np.complex128)
    #
    # hxy = np.zeros((1, ntaps), dtype=np.complex128)
    # hyx = np.zeros((1, ntaps), dtype=np.complex128)
    # hyy = np.zeros((1, ntaps), dtype=np.complex128)
    #
    # hxx[0, ntaps // 2] = 1
    # hyy[0, ntaps // 2] = 1

    hxx = np.zeros((1, ntaps), dtype=np.complex128)

    hxy = np.zeros((1, ntaps), dtype=np.complex128)
    hyx = np.zeros((1, ntaps), dtype=np.complex128)
    hyy = np.zeros((1, ntaps), dtype=np.complex128)

    hyx[0, ntaps // 2] = 1
    hxy[0, ntaps // 2] = 1
    return (hxx, hxy, hyx, hyy)


def demap_to_msg(rx_symbols, order, do_normal=True):
    '''

    :param rx_symbols:1d array or 2d array if 2d the shape[0] must be 1
    :param do_normal: if True the rx_symbols will be normalized to 1
    :return: 2d array msg, the shape[0] is 1
    '''
    from scipy.io import loadmat

    base_path = os.path.abspath(__file__)
    base_path = os.path.dirname(os.path.dirname(base_path))

    if order == 8:
        raise NotImplementedError('Not implemented yet')

    constl = cal_symbols_qam(order) / np.sqrt(cal_scaling_factor_qam(order))
    rx_symbols = np.atleast_2d(rx_symbols)
    assert rx_symbols.shape[0] == 1
    rx_symbols = rx_symbols[0]

    qam_data = loadmat(f'{base_path}/qamdata/{order}qam.mat')['x'][0]
    msg = np.zeros((1, rx_symbols.shape[0]))
    if do_normal:
        rx_symbols = rx_symbols / np.sqrt(np.mean(rx_symbols.imag ** 2 + rx_symbols.real ** 2))

    for index, symbol in enumerate(rx_symbols):
        decision_symbol = decision(symbol, constl=np.atleast_2d(constl))

        choose = np.abs(decision_symbol - qam_data) < np.spacing(1)
        #         print(choose)
        #         print(np.nonzero(choose)[0])
        msg[0, index] = (np.nonzero(choose)[0])
    #         print(msg[0,index])
    #         break

    return msg.astype(np.int)[0,:]


def main():
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    cc = loadmat('test2.mat')
    tx = cc['txsymbols']
    sam = cc['rxSignalIn']

    import time

    now = time.time()

    constl = cal_symbols_qam(16)
    constl = constl / np.sqrt(cal_scaling_factor_qam(16))
    constl = np.atleast_2d(constl)
    xsym, ysym, wer, errorsx, errorsy = dual_pol_time_domain_lms_equalizer(sam[:], 13, 2, constl=constl,
                                                                           train_symbol=tx,
                                                                           mu=0.001, niter=3)
    plt.plot(errorsx)
    print(time.time() - now)


if __name__ == '__main__':
    main()
