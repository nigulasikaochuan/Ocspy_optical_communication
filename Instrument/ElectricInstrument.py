from scipy.signal import convolve, fftconvolve, resample_poly
from scipy.signal import resample
import numpy as np


class ADC(object):

    def __call__(self, signal):
        # print("ad will be implemented")
        # print("after ad, the sample in fiber will be resampled to signal.sps")
        # print("and the signal.sps_in_fiber will be set to signal.sps")

        sps_in_fiber = signal.sps_in_fiber
        sps = signal.sps

        N = 1/(sps / sps_in_fiber)
        N = int(N)

        tempx = resample_poly(signal[0], 1, N)
        tempy = resample_poly(signal[1], 1, N)

        new_sample = np.array([tempx, tempy])
        signal.data_sample_in_fiber = new_sample
        signal.sps_in_fiber = signal.sps
        return signal


class DAC(object):

    def __call__(self, signal):
        sps_in_fiber = np.ceil(signal.sps_in_fiber)
        N = np.ceil(sps_in_fiber / signal.sps)
        N = int(N)
        from scipy.signal import resample_poly
        # tempx = resample(signal.data_sample[0, :], N)
        # tempy = resample(signal.data_sample[1, :], N)
        tempx = resample_poly(signal.data_sample, up=N, down=1, axis=1)
        signal.data_sample_in_fiber = tempx
        # signal.sps_in_fiber = sps_in_fiber


class PulseShaping(object):
    '''
        Perform Pluse Shaping. need a dict to construct, the construct should contain:
            key:pulse_shaping
                span
                sps
                alpha

    '''

    def __init__(self, **kwargs):
        self.kind = kwargs['kind']
        self.span = kwargs['span']
        self.sps = kwargs['sps']
        self.alpha = kwargs['alpha']
        assert divmod(self.span * self.sps, 2)[1] == 0
        self.number_of_sample = self.span * self.sps
        self.delay = self.span / 2 * self.sps

        self.filter_tap = self.__design_filter()

    def __design_filter(self):
        if self.kind == 'rrc':
            h = PulseShaping.rcosdesign(
                self.number_of_sample, self.alpha, 1, self.sps)
            return np.atleast_2d(h)

        if self.kind == 'rc':
            print(
                'error, why do you want to design rc filter,the practical use should be two rrc filters,on in transimit'
                'side,and one in receiver side, rrc filter will be designed')

    @staticmethod
    def rcosdesign(N, alpha, Ts, Fs):
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

    def rrcfilter(self, signal_interface):
        '''

        :param signal_interface: signal object to be pulse shaping,because a reference of signal object is passed
        so the filter is in place
        :return: None

        '''

        # print("---begin pulseshaping ---,the data sample will be set")
        # upsample by insert zeros

        signal_interface.data_sample = np.zeros(
            (signal_interface.symbol.shape[0], signal_interface.sps * signal_interface.symbol_length))
        signal_interface.data_sample = np.asarray(
            signal_interface.data_sample, dtype=np.complex)
        for i in range(signal_interface.symbol.shape[0]):
            signal_interface.data_sample[i, :] = signal_interface.upsample(signal_interface.symbol[i, :],
                                                                           signal_interface.sps)[0, :]

        temp = []
        for i in range(signal_interface.data_sample.shape[0]):
            temp.append(
                fftconvolve( signal_interface.data_sample[i, :],self.filter_tap[0, :],mode='full'))

        # tempy = convolve(self.filter_tap[0, :], signal_interface.data_sample[1, :])
        # temp_signal = np.array([tempx, tempy])
        temp_signal = np.array(temp)
        # compensate group delay
        temp_signal = np.roll(temp_signal, -int(self.delay), axis=1)

        signal_interface.data_sample = temp_signal[:,
                                                   :signal_interface.sps * signal_interface.symbol_length]

    def __call__(self, signal):
        self.rrcfilter(signal)


class AWG(object):

    def __init__(self, sps=2, alpha=0.02, span=1024):
        '''

        :param sps: pulse shaping parameters
        :param alpha: reference of the signal object,change signal_interface's property will change all
        :param span

        This function will be used to pulse shaping, the sps is default to 2, it should correspond to signal's sps, then
        the siganl will be resampled to sps_in_fiber automatically

        '''

        # self.signal_interface = signal_interface
        self.roll_off = alpha
        self.span = span
        self.pulse_shaping_filter = PulseShaping(
            kind='rrc', alpha=alpha, sps=sps, span=span)
        self.dac = DAC()

    def __call__(self, signal_interface):
        self.pulse_shaping_filter.rrcfilter(signal_interface)

        self.dac(signal_interface)
        return signal_interface

    def __str__(self):
        string = f"the pulse shaping filter: roll_off:{self.pulse_shaping_filter.alpha}," \
            f" the sps of pulse shaping filter : {self.signal_interface.sps},  the adc sample per symbol: {self.signal_interface.sps_in_fiber} "

        return string

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    # h = PulseShaping.rcosdesign(32 * 4, 0.2, 1, 4)
    import progressbar
    import time

    bar = progressbar.ProgressBar(max_value=16000)
    for i in range(16000):
        bar.update(i + 1)
        time.sleep(0.01)

    bar.update(16000)
    bar.finish()
