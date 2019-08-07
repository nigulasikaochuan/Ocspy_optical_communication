#%%
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
from dsp_tools import normal_sample, decision
from dsp import MatchedFilter,dual_pol_time_domain_lms_equalizer,syncsignal,superscalar
import numpy as np
import matplotlib.pyplot as plt
from dsp_tools import cal_symbols_qam,cal_scaling_factor_qam


#%%




def get_signal(nch,power):

    signals = generate_signal(baudrates=35e9,nch=nch,powers = power,grid_size=50e9,mf='16qam')
    wdm = mux_signal(signals)
    return signals,wdm

# for power in range(1,4):
#     nch = 7
#
#     signals,wdm = get_signal(nch,1)
#     spans = NonlinearFiber(0.2,16.7,80,1.3,step_length=20/1000)
#     edfas = Edfa(gain_db=16,nf=5,is_ase=False)
#
#     span_number = 10
#     for i in tqdm_notebook(range(span_number)):
#
#         wdm = spans(wdm)
#         wdm = edfas(wdm)
#
#     import joblib
#     joblib.dump(dict(wdm=wdm,center_signal = signals[nch//2]),f'data_qpsk{power}')
#

def demodulate(wdm,spans,nch,center_signal_ori):
    from dsp import cd_compensation
    cons = cal_symbols_qam(16) / np.sqrt(cal_scaling_factor_qam(16))

    wdm = cd_compensation(wdm,spans,False)
    center_signal = demux_signal(wdm, nch // 2)
    pos_freq = center_signal_ori.baudrate / 2 * (1 + 0.02)
    neg_freq = - pos_freq
    center_signal = ideal_lowpass(center_signal, pos_freq, neg_freq, wdm.fs_in_fiber)
    center_signal = ADC()(center_signal, sps_in_fiber=center_signal_ori.sps_in_fiber, sps=2)
    center_signal = MatchedFilter(0.02, 2)(center_signal)
    center_signal = normal_sample(center_signal)
    # center_signal = center_signal[:,::2]

    center_signal = dual_pol_time_domain_lms_equalizer(center_signal, 321, 2, np.atleast_2d(cons),center_signal_ori.symbol,
                                                       mu=0.001,niter=3)

    xpol = center_signal[0]
    ypol = center_signal[1]

    xpol = syncsignal(center_signal_ori.symbol[0], xpol, 1)
    ypol = syncsignal(center_signal_ori.symbol[1], ypol, 1)

    xpol, xpol_ori = superscalar(xpol, center_signal_ori.symbol[0], 200, 4, cons, 0.02)
    ypol, ypol_ori = superscalar(ypol, center_signal_ori.symbol[1], 200, 4, cons, 0.02)
    # #
    mask = xpol!=0
    xpol  = xpol[mask][np.newaxis,:]
    ypol = ypol[mask][np.newaxis,:]
    # scatterplot(xpol,1)
    xpol_ori = xpol_ori[:,mask[0]]
    ypol_ori = ypol_ori[:,mask[0]]
    # #
    noise = xpol-xpol_ori[0]
    power = power_meter(noise,'w')
    #
    print(np.log10(1/power)*10)


    c0 = np.abs(xpol[0]) -np.mean(np.abs(xpol[0]))
    c0 = correlate(c0,c0.conj())

    return c0,xpol,ypol,cons



#%%
import joblib

name = ['data0','data1','data2','data3']
c11 = []
c12 = []
c13 = []

for n in name:
    wdm = joblib.load(n)['wdm']
    center_signal_ori = joblib.load(n)['center_signal']
    print(power_meter(center_signal_ori,'dbm'))
    spans = [NonlinearFiber(0.2,16.7,80,1.3,step_length=20/1000)]*10

    c0,xpol,ypol,cons= demodulate(wdm, spans, 7, center_signal_ori)
    # max = np.argmax(c0)
    # plt.plot((c0[max+1:max+1+600]))
    # plt.show()

    distance = np.unique(np.abs(cons))

    circle1 = cons[np.abs(cons)==distance[0]]
    circle2 = cons[np.abs(cons)==distance[1]]
    circle3 = cons[np.abs(cons)==distance[2]]

    xpol2 = []

    circle1_demod = []
    circle2_demod = []
    circle3_demod = []

    circle11_demod = xpol[:,np.abs(np.abs(xpol[0])-distance[0])<0.1]
    circle12_demod = xpol[:,np.abs(np.abs(xpol[0])-distance[1])<0.1]
    circle13_demod = xpol[:,np.abs(np.abs(xpol[0])-distance[2])<0.1]

    c11_power0 = np.abs(circle11_demod[0]) - np.mean(np.abs(circle11_demod[0]))
    c12_power0 = np.abs(circle12_demod[0]) - np.mean(np.abs(circle12_demod[0]))
    c13_power0 = np.abs(circle13_demod[0]) - np.mean(np.abs(circle13_demod[0]))
    c11_power0 = correlate(c11_power0, c11_power0.conj())
    c12_power0 = correlate(c12_power0, c12_power0.conj())
    c13_power0 = correlate(c13_power0, c13_power0.conj())

    max1 = np.argmax(c11_power0)
    max2 = np.argmax(c12_power0)

    max3 = np.argmax(c13_power0)
    c11.append(c11_power0[max1+1:max1+1+160])
    c12.append(c12_power0[max2+1:max2+1+160])
    c13.append(c13_power0[max3+1:max3+1+160])


    #
    # max = np.argmax(c13_power0)
    # plt.plot((c13_power0[max+1:max+1+16]))
    # plt.show()


# max = np.argmax(c0)
# plt.plot((c0[max+
#
# circle2 = circle2 - (circle2-circle1)
#
# scatterplot(circle2[np.newaxis,:],1)
#
#

#%%
# plt.plot(c11[0],label='0dbm')
# plt.plot(c11[1],label='1dbm')
# plt.plot(c11[2],label='2dbm')
# plt.plot(c11[3],label='3dbm')
# plt.legend()
# plt.show()
#
# #%%
#
# plt.plot(c12[0],label='0dbm')
# plt.plot(c12[1],label='1dbm')
# plt.plot(c12[2],label='2dbm')
# plt.plot(c12[3],label='3dbm')
# plt.legend()
# plt.show()
#
# #%%
# plt.plot(c13[0],label='0dbm')
# plt.plot(c13[1],label='1dbm')
# plt.plot(c13[2],label='2dbm')
# plt.plot(c13[3],label='3dbm')
# plt.legend()
# plt.show()

#%%
corr0dbm = ((c11[0]+c12[0]+c13[0]))/3

corr1dbm = ((c11[1]+c12[1]+c13[1]))/3
corr2dbm = ((c11[2]+c12[2]+c13[2]))/3
corr3dbm = ((c11[3]+c12[3]+c13[3]))/3
plt.plot(corr0dbm,label='0dbm')
plt.plot(corr1dbm,label='1dbm')
plt.plot(corr2dbm,label='2dbm')
plt.plot(corr3dbm,label='3dbm')
plt.legend()
plt.show()

#%%
 # confirm only ase
# nch = 7
# power = 0
# signals, wdm = get_signal(nch, power)
#
# snr = 16
# noise = power_meter(wdm,'w')/(10**(snr/10)) *20 /7
#
# noise_seq = np.random.randn(*wdm.shape)+1j*np.random.randn(*wdm.shape)
# noise_seq = np.sqrt(noise/2/2)*noise_seq
# wdm[:]  = wdm[:] + noise_seq
# spans = [NonlinearFiber(0.2, 16.7, 80, 1.3, step_length=20 / 1000)] * 10
# center_signal_ori =  signals[nch//2]
# c0,xpol,ypol,cons= demodulate(wdm, spans, nch, center_signal_ori)
#
# # scatterplot(xpol,1)
#
# distance = np.unique(np.abs(cons))
#
# circle1 = cons[np.abs(cons)==distance[0]]
# circle2 = cons[np.abs(cons)==distance[1]]
# circle3 = cons[np.abs(cons)==distance[2]]
#
# xpol2 = []
#
# circle1_demod = []
# circle2_demod = []
# circle3_demod = []
#
# circle11_demod = xpol[:,np.abs(np.abs(xpol[0])-distance[0])<0.1]
# circle12_demod = xpol[:,np.abs(np.abs(xpol[0])-distance[1])<0.1]
# circle13_demod = xpol[:,np.abs(np.abs(xpol[0])-distance[2])<0.1]
#
# c11_power0 = np.abs(circle11_demod[0]) - np.mean(np.abs(circle11_demod[0]))
# c12_power0 = np.abs(circle12_demod[0]) - np.mean(np.abs(circle12_demod[0]))
# c13_power0 = np.abs(circle13_demod[0]) - np.mean(np.abs(circle13_demod[0]))
# c11_power0 = correlate(c11_power0, c11_power0.conj())
# c12_power0 = correlate(c12_power0, c12_power0.conj())
# c13_power0 = correlate(c13_power0, c13_power0.conj())
# max1 = np.argmax(c11_power0)
# max2 = np.argmax(c12_power0)
# max3 = np.argmax(c13_power0)
#
# corr1dbm = (c11_power0[max1+1:max1+1+260]+c12_power0[max2+1:max2+1+260]+c13_power0[max3+1:max3+1+260])/3
# #
#
# plt.plot(corr1dbm,label='3dbm')
