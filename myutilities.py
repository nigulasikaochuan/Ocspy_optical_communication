from ElecInstrument import PulseShaping, DAC
import math
import Signal
import numpy as np
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


def generate_signal(nch, powers, baudrates, grid_size, start_freq=193.1e12,alpha=0.02,mf='16qam'):
    if not isinstance(baudrates, list):
        baudrates = [baudrates] * nch
    else:
        assert len(baudrates) == nch
    if not isinstance(powers, list):
        powers = [powers] * nch
    else:
        assert len(powers) == nch

    sps_in_fiber = calc_sps_in_fiber(nch, grid_size, baudrates[0])
    shaping_filter = PulseShaping(alpha=alpha, span=1024, sps=2)
    freqs = [start_freq + i * grid_size for i in range(nch)]

    signals = []
    for freq, power, baudrate in zip(freqs, powers, baudrates):
        config = dict(baudrate=baudrate, sps_in_fiber=sps_in_fiber, unit='hz', unit_freq='hz', center_frequency=freq,
                        launch_power = power,
                      mf=mf)
        signals.append(Signal.QamSignal(**config))

    for signal in signals:
        signal = shaping_filter(signal)
        signal = DAC(is_quanti=False, clipping_ratio=None, resolution_bits=None)(signal)

    for index, signal in enumerate(signals):
        print(index, end=' ')
        normal_sample(signal[:])
        power = 10 ** (signal.launch_power / 10) / 1000 / 2
        signal[:] = np.sqrt(power) * signal[:]

    return signals
