import copy

import numpy as np
from numpy.fft import fftfreq
from scipy.fftpack import fft, ifft

from Signal import Signal


def rrcfilter(alpha,span,sps):
    assert divmod(span * sps, 2)[1] == 0

    return  _rcosdesign(span*sps, alpha, 1, sps)

def _rcosdesign(N, alpha, Ts, Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.
    Parameters
    ----------
    N : int
        Length of the filter in samples.
    alpha : float
        Roll off factor (Valid values are [0, 1]).
    Ts : float
        Symbol period in seconds.
    Fs : float
        Sampling Rate in Hz.
    Returns
    ---------
    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.
    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.
    """

    T_delta = 1 / float(Fs)

    sample_num = np.arange(N + 1)
    h_rrc = np.zeros(N + 1, dtype=float)

    for x in sample_num:
        t = (x - N / 2) * T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and t == Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) *
                                                (np.sin(np.pi / (4 * alpha)))) + (
                                                       (1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
        elif alpha != 0 and t == -Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) *
                                                (np.sin(np.pi / (4 * alpha)))) + (
                                                       (1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi * t * (1 - alpha) / Ts) +
                        4 * alpha * (t / Ts) * np.cos(np.pi * t * (1 + alpha) / Ts)) / \
                       (np.pi * t * (1 - (4 * alpha * t / Ts)
                                     * (4 * alpha * t / Ts)) / Ts)

    return h_rrc / np.sqrt(np.sum(h_rrc * h_rrc))



def ideal_lowpass(samples, pos_cutoff_frequence, neg_cutoff_frequence, fs):


    samples = np.atleast_2d(samples)

    freq = fftfreq(samples.shape[1], 1 / fs)

    fft_sample = fft(samples, axis=1)
        # 超过滤波器带宽的频率点直接设置为0
    for i in range(samples.shape[0]):
        fft_sample[i, freq > pos_cutoff_frequence] = 0
        fft_sample[i, freq < neg_cutoff_frequence] = 0

    samples = ifft(fft_sample, axis=1)

    return samples