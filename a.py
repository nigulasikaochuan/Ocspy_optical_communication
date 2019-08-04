from scipy.signal import correlate

from Signal import QamSignal
from myutilities import generate_signal
from OpticalInstrument import mux_signal
from OpticalInstrument import demux_signal
from OpticalInstrument import Edfa
from channel import NonlinearFiber
from tools import spectrum_analyzer, scatterplot, power_meter
from Filter import ideal_lowpass
from ElecInstrument import ADC
from dsp_tools import normal_sample
from dsp import MatchedFilter,dual_pol_time_domain_lms_equalizer,syncsignal,superscalar
import numpy as np
import matplotlib.pyplot as plt
from dsp_tools import cal_symbols_qam,cal_scaling_factor_qam

def get_signal(nch,power):

    signals = generate_signal(baudrates=35e9,nch=nch,powers = power,grid_size=50e9)
    wdm = mux_signal(signals)
    return signals,wdm

nch = 7
power = 0
signals,wdm = get_signal(nch,power)
spans = NonlinearFiber(0.2,16.7,80,1.3,step_length=20/1000)
edfas = Edfa(gain_db=16,nf=5,is_ase=False)

span_number = 10
for i in range(span_number):
    wdm = spans(wdm)
    wdm = edfas(wdm)

import joblib
joblib.dump(dict(wdm=wdm,center_signal = signals[nch//2]),f'data{power}')





# cons = cal_symbols_qam(16) / np.sqrt(cal_scaling_factor_qam(16))
# center_signal = demux_signal(wdm, nch // 2)
#
# pos_freq = signals[nch // 2].baudrate / 2 * (1 + 0.02)
# neg_freq = - pos_freq
# center_signal = ideal_lowpass(center_signal, pos_freq, neg_freq, wdm.fs_in_fiber)
# center_signal = ADC()(center_signal, sps_in_fiber=signals[nch // 2].sps_in_fiber, sps=2)
# center_signal = MatchedFilter(0.02, 2)(center_signal)
# center_signal = normal_sample(center_signal)
#
# center_signal = dual_pol_time_domain_lms_equalizer(center_signal, 321, 2, np.atleast_2d(cons), signals[nch // 2].symbol,
#                                                    mu=0.001)
# xpol = center_signal[0]
# ypol = center_signal[1]
# print('hello world')
#
# xpol = syncsignal(signals[nch // 2].symbol[0], xpol, 1)
# ypol = syncsignal(signals[nch // 2].symbol[1], ypol, 1)

