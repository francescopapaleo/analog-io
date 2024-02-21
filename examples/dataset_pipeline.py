"""Analog I/O Tools for Python
Copyright (C) 2024 Francesco Papaleo

Dataset pipeline example
"""

import numpy as np
import os
import soundfile as sf
from analog_io import SignalGenerator


def frequency_to_midi(frequency):
    """Convert frequency to MIDI note number."""
    return 12 * np.log2(frequency / 440.0) + 69


# Constants
SAMPLE_RATE = 48000
BIT_DEPTH = "PCM_24"
TOT_DURATION = 5
SIG_DURATION = 0.5
IMPULSE_START = 0.5
DESTINATION_FOLDER = "/Users/fra/Desktop/sandbox/dry"
PEAK_DBFS = -9

# Pitch calculation
start_pitch = 32.70
end_pitch = 4186.01
pitch_step = 2 ** (1 / 12)
num_pitches = int(np.log(end_pitch / start_pitch) / np.log(pitch_step)) + 1
pitches = [start_pitch * (pitch_step**n) for n in range(num_pitches)]

# SignalGenerator instance
sg = SignalGenerator(sample_rate=SAMPLE_RATE, bit_depth=BIT_DEPTH, channel_mode="mono")

# Ensure destination folder exists
if not os.path.exists(DESTINATION_FOLDER):
    os.makedirs(DESTINATION_FOLDER)

waveforms = ["sin", "sqr", "saw"]

# Generate and save files
for waveform in waveforms:
    for pitch in pitches:
        midi_note = int(
            round(frequency_to_midi(pitch))
        )  # Convert frequency to MIDI note and round
        # Correctly generate tone based on waveform
        audio = sg.generate_tone(
            waveform, pitch, SIG_DURATION, PEAK_DBFS, IMPULSE_START, TOT_DURATION
        )

        # Filename construction with MIDI note number
        filename = f"dry_{waveform}_{midi_note}.wav"  # Use MIDI note number in filename
        sf.write(
            os.path.join(DESTINATION_FOLDER, filename), audio, SAMPLE_RATE, BIT_DEPTH
        )

print(f"Generated {len(pitches) * len(waveforms)} files in '{DESTINATION_FOLDER}'.")

from analog_io import HardwareLatencyMeasure

tester = HardwareLatencyMeasure()
tester.select_device_and_channels()
latency_time, latency_index = tester.measure_latency()

from analog_io import AnalogProcessor

processor = AnalogProcessor(
    input_folder="/Users/fra/Desktop/sandbox/dry",
    output_folder="/Users/fra/Desktop/sandbox/wet",
    sample_rate=48000,
    device_index=6,
    input_channel_index=1,
    output_channel_index=3,
    latency_samples=latency_index,
    wait_time=4.0,
)

processor.process_files()
