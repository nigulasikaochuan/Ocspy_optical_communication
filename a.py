import Signal
import numpy as np
from Signal import QamSignal
from channel import NonlinearFiber
from ElecInstrument import PulseShaping,DAC
from OpticalInstrument import mux_signal
from tools import power_meter, spectrum_analyzer
import math
from tqdm import tqdm_notebook


def calc_sps_in_fiber(nch, spacing, baudrate):
    if divmod(nch, 2)[1] == 0:
        highest = nch / 2 * spacing * 4
        sps_in_fiber = math.ceil(highest / spacing)
    else:
        highest = 4 * ((nch - 1) / 2 * spacing + spacing / 2)
        sps_in_fiber = math.ceil(highest / baudrate)
    if divmod(sps_in_fiber, 2)[1] != 0:
        sps_in_fiber += 1
    return sps_in_fiber


def normal_sample(signal_samples):
    signal_samples = np.atleast_2d(signal_samples)

    for i in range(signal_samples.shape[0]):
        signal_samples[i] = signal_samples[i] / np.sqrt(np.mean(np.abs(signal_samples[i]) ** 2))

    return signal_samples


def generate_signal(nch, power, baudrate, grid_size, start_freq=193.1e12):
    if not isinstance(baudrate, list):
        baudrates = [baudrate] * nch
    else:
        assert len(baudrate) == nch
    if not isinstance(power, list):
        powers = [power] * nch
    else:
        assert len(power) == nch

    sps_in_fiber = calc_sps_in_fiber(nch, grid_size, baudrate)
    shaping_filter = PulseShaping(alpha=0.02, span=1024, sps=2)
    freqs = [start_freq + i * grid_size for i in range(nch)]

    signals = []
    for freq, power, baudrate in zip(freqs, powers, baudrates):
        config = dict(baudrate=baudrate, sps_in_fiber=sps_in_fiber, unit='hz', unit_freq='hz', center_frequency=freq)
        signals.append(QamSignal(**config))

    for signal in signals:
        signal = shaping_filter(signal)
        signal = DAC(is_quanti=False, clipping_ratio=None, resolution_bits=None)(signal)

    for index, signal in enumerate(signals):
        print(index, end=' ')
        normal_sample(signal[:])
        power = 10 ** (signal.launch_power / 10) / 1000 / 2
        signal[:] = np.sqrt(power) * signal[:]

    return signals

c = generate_signal(40,0,34.1486e9,37.5e9)
wdm_signal = mux_signal(c)
spectrum_analyzer(wdm_signal,fs=wdm_signal.fs_in_fiber)


# spectrum_analyzer(wdm_signal)
