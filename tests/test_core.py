"""# Analog I/O Tools for Python
Copyright (C) 2024 Francesco Papaleo
"""

import pytest
import numpy as np
from analog_io import SignalGenerator, HardwareLatencyMeasure, dbfs_to_amp


def test_dbfs_to_linear():
    assert dbfs_to_amp(0) == 1
    assert dbfs_to_amp(-6) == pytest.approx(0.501187, 0.0001)
    assert dbfs_to_amp(-20) == 0.1


def test_generate_time_array():
    sg = SignalGenerator(sample_rate=1000)
    time_array = sg.generate_time_array(1)
    assert len(time_array) == 1000
    assert np.allclose(time_array, np.linspace(0, 1, 1000, endpoint=False))


@pytest.mark.parametrize("waveform", ["sin", "saw", "sqr"])
def test_waveform_generation(waveform):
    sg = SignalGenerator(sample_rate=1000)
    signal = getattr(sg, f"{waveform}")(100, 1, -3)
    assert len(signal) == 1000


def test_adjust_channels():
    sg = SignalGenerator(sample_rate=1000, channel_mode="stereo")
    signal = np.ones(1000)
    adjusted_signal = sg.mono_to_stereo(signal)
    assert adjusted_signal.shape == (1000, 2)
    assert np.all(adjusted_signal == np.stack((signal, signal), axis=-1))


def test_place_signal():
    sg = SignalGenerator(sample_rate=1000)
    signal = np.ones(100)
    placed_signal = sg.place_signal(signal, 1, 3)
    assert len(placed_signal) == 3000
    assert np.all(placed_signal[1000:1100] == 1)
    assert np.all(placed_signal[:1000] == 0)
    assert np.all(placed_signal[1100:] == 0)


def test_generate_tone():
    sg = SignalGenerator(sample_rate=1000)
    tone = sg.generate_tone("sin", 100, 1, -3, 1, 3)
    assert len(tone) == 3000


def test_hardware_latency_measure_init():
    hlm = HardwareLatencyMeasure()
    assert hlm.sample_rate == 48000
    assert hlm.duration == 5
    assert hlm.start_time == 2.5
    assert hlm.pulse_width == 0.001
    assert hlm.amplitude_dbfs == -1


def test_generate_pulse_signal():
    hlm = HardwareLatencyMeasure(
        sample_rate=1000, duration=3, pulse_width=0.001, amplitude_dbfs=0
    )
    pulse_signal = hlm.generate_pulse_signal()
    assert len(pulse_signal) == 3000
    assert np.sum(pulse_signal) == 1


def test_find_delay():
    hlm = HardwareLatencyMeasure(sample_rate=1000)
    original = np.zeros(3000)
    original[1000:1100] = 1
    recorded = np.zeros(3000)
    recorded[1200:1300] = 1
    delay_time, delay_index = hlm.find_delay(original, recorded)
    assert delay_time == 0.2
    assert delay_index == 200
