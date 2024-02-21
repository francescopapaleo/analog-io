"""Signal Generator Package
Copyright (C) 2024 Francesco Papaleo
"""

import pytest
import numpy as np
from signal_generator import SignalGenerator

def test_dbfs_to_amplitude():
    sg = SignalGenerator()
    assert sg.dbfs_to_amplitude(0) == 1
    assert sg.dbfs_to_amplitude(-6) == pytest.approx(0.501187, 0.0001)
    assert sg.dbfs_to_amplitude(-20) == 0.1

def test_place_signal():
    sg = SignalGenerator(sample_rate=1000)
    signal = np.ones(100)
    full_signal = sg.place_signal(signal, start_time=1, signal_duration=0.1, total_duration=3)
    assert len(full_signal) == 3000
    assert np.sum(full_signal) == 100

def test_generate_impulse():
    sg = SignalGenerator(sample_rate=1000)
    impulse = sg.generate_impulse(start_time=1, signal_duration=0.001, total_duration=3, pulse_amplitude_dbfs=0)
    assert len(impulse) == 3000
    assert np.sum(impulse) == 1

def test_generate_tone():
    sg = SignalGenerator(sample_rate=1000)
    tone = sg.generate_tone(pitch=440, signal_duration=1, total_duration=3, waveform='sin', start_time=1, max_amplitude_dbfs=0)
    assert len(tone) == 3000
    assert np.sum(tone) != 0

def test_generate_chirp():
    sg = SignalGenerator(sample_rate=1000)
    chirp = sg.generate_chirp(f0=200, f1=600, signal_duration=1, total_duration=3, method="logarithmic", start_time=1, max_amplitude_dbfs=0)
    assert len(chirp) == 3000
    assert np.sum(chirp) != 0

def test_generate_log_sweep():
    sg = SignalGenerator(sample_rate=1000)
    sweep = sg.generate_log_sweep(f0=20, f1=20000, signal_duration=1, total_duration=3, inverse=False, amplitude_dbfs=0, start_time=1)
    assert len(sweep) == 3000
    assert np.sum(sweep) != 0