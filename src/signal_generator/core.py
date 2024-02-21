"""Signal Generator Package
Copyright (C) 2024 Francesco Papaleo
"""

import numpy as np
import soundfile as sf
import os
from scipy import signal


class SignalGenerator:
    """
    A class used to generate different types of audio signals.

    Parameters
    ----------
    sample_rate : int, optional
        The sample rate of the audio signal (default is 48000).
    bit_depth : str, optional
        The bit depth of the audio signal (default is 'PCM_24').
    channel_mode : str, optional
        The channel mode of the audio signal (default is 'mono').
    waveforms : list of str, optional
        The types of waveforms that can be generated (default is ['sin', 'sqr', 'saw', 'chirp', 'log_sweep']).
    """
    def __init__(self, sample_rate=48000, bit_depth='PCM_24', channel_mode='mono', waveforms=['sin', 'sqr', 'saw', 'chirp', 'log_sweep']):
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.channel_mode = channel_mode
        self.waveforms = waveforms

    def dbfs_to_amplitude(self, dbfs):
        """
        Convert dBFS to a linear amplitude scale.

        Parameters
        ----------
        dbfs : float
            The dBFS value to convert.

        Returns
        -------
        float
            The converted amplitude value.
        """
        return 10 ** (dbfs / 20)

    def place_signal(self, signal, start_time, signal_duration, total_duration):
        """
        Place the signal at start_time within a total_duration, ensuring the signal fits within the specified durations.

        Parameters
        ----------
        signal : ndarray
            The signal to place.
        start_time : float
            The start time to place the signal within total_duration.
        signal_duration : float
            The duration of the signal.
        total_duration : float
            The total duration for the resulting signal including silence before and after the signal.

        Returns
        -------
        ndarray
            The full signal with silence before and after to fit within total_duration.
        """
        total_samples = int(self.sample_rate * total_duration)
        signal_samples = int(self.sample_rate * signal_duration)
        start_sample = int(self.sample_rate * start_time)
        
        # Ensure the generated signal matches the signal_duration before placement
        if len(signal) > signal_samples:
            signal = signal[:signal_samples]
        elif len(signal) < signal_samples:
            # Extend the signal with silence if it's shorter than expected
            signal = np.pad(signal, (0, signal_samples - len(signal)), 'constant', constant_values=(0, 0))

        silence_before = np.zeros(start_sample)
        end_sample = start_sample + len(signal)
        silence_after_length = total_samples - end_sample
        silence_after = np.zeros(max(0, silence_after_length))  # Ensure no negative length

        full_signal = np.concatenate([silence_before, signal, silence_after])
        return full_signal

    def generate_impulse(self, start_time=0, signal_duration=0.001, total_duration=5, pulse_amplitude_dbfs=-1):
        """
        Generate an impulse signal and place it within a total_duration.

        Parameters
        ----------
        start_time : float, optional
            The start time of the impulse within the total_duration (default is 0).
        signal_duration : float, optional
            The duration of the impulse (default is 0.001).
        total_duration : float, optional
            The total duration of the resulting signal including silence (default is 5).
        pulse_amplitude_dbfs : float, optional
            The amplitude of the pulse in dBFS (default is -1).

        Returns
        -------
        ndarray
            The generated and placed impulse signal.
        """
        pulse_amplitude = self.dbfs_to_amplitude(pulse_amplitude_dbfs)
        impulse = np.zeros(int(self.sample_rate * signal_duration))
        impulse[:] = pulse_amplitude
        placed_impulse = self.place_signal(impulse, start_time, signal_duration, total_duration)
        if self.channel_mode == 'stereo':
            placed_impulse = np.tile(placed_impulse[:, np.newaxis], (1, 2))
        return placed_impulse


    def generate_tone(self, pitch, signal_duration=5, total_duration=5, waveform='sin', start_time=0, max_amplitude_dbfs=-1):
        """
        Generate a tone signal of a specified waveform and pitch.

        Parameters
        ----------
        pitch : float
            The frequency of the tone in Hz.
        signal_duration : float, optional
            The duration of the tone in seconds (default is 5).
        total_duration : float, optional
            The total duration of the resulting signal including silence (default is 5).
        waveform : str, optional
            The type of waveform ('sin' for sine wave, 'sqr' for square wave, 'saw' for sawtooth wave; default is 'sin').
        start_time : float, optional
            The start time of the tone within the duration (default is 0).
        max_amplitude_dbfs : float, optional
            The maximum amplitude of the tone in dBFS (default is -1).

        Returns
        -------
        ndarray
            The generated tone signal.
        """
        amplitude = self.dbfs_to_amplitude(max_amplitude_dbfs)
        t = np.linspace(0, signal_duration, int(signal_duration * self.sample_rate))
        if waveform == 'sin':
            tone_signal = np.sin(2 * np.pi * pitch * t) * amplitude
        elif waveform == 'sqr':
            tone_signal = signal.square(2 * np.pi * pitch * t) * amplitude
        elif waveform == 'saw':
            tone_signal = signal.sawtooth(2 * np.pi * pitch * t) * amplitude

        # Place the generated signal at the specified start time
        placed_signal = self.place_signal(tone_signal, start_time, signal_duration, total_duration)
        if self.channel_mode == 'stereo':
            placed_signal = np.tile(placed_signal[:, np.newaxis], (1, 2))
        return placed_signal


    def generate_chirp(self, f0=200, f1=600, signal_duration=2, total_duration=5, method="logarithmic", start_time=0, max_amplitude_dbfs=-1):
        """
        Generate a chirp signal that sweeps from one frequency to another.

        Parameters
        ----------
        f0 : float, optional
            The starting frequency of the chirp in Hz (default is 200).
        f1 : float, optional
            The ending frequency of the chirp in Hz (default is 600).
        signal_duration : float, optional
            The duration of the chirp in seconds (default is 2).
        total_duration : float, optional
            The total duration of the resulting signal including silence (default is 5).
        method : str, optional
            The method of frequency sweep ('linear' or 'logarithmic'; default is 'logarithmic').
        start_time : float, optional
            The start time of the chirp within a total duration of 5 seconds (default is 0).
        max_amplitude_dbfs : float, optional
            The maximum amplitude of the chirp in dBFS (default is -1).

        Returns
        -------
        ndarray
            The generated chirp signal.
        """
        amplitude = self.dbfs_to_amplitude(max_amplitude_dbfs)
        t = np.linspace(0, signal_duration, int(signal_duration * self.sample_rate))
        chirp_signal = signal.chirp(t, f0=f0, t1=signal_duration, f1=f1, method=method) * amplitude

        # Place the generated chirp signal at the specified start time
        placed_chirp = self.place_signal(chirp_signal, start_time, signal_duration, total_duration)
        if self.channel_mode == 'stereo':
            placed_chirp = np.tile(placed_chirp[:, np.newaxis], (1, 2))
        return placed_chirp

    def generate_log_sweep(self, f0=20, f1=20000, signal_duration=5, total_duration=5, inverse=False, amplitude_dbfs=-1, start_time=0):
        """
        Generate a logarithmic sweep tone or its inverse.

        Parameters
        ----------
        f0 : float, optional
            The starting frequency of the sweep in Hz (default is 20).
        f1 : float, optional
            The ending frequency of the sweep in Hz (default is 20000).
        signal_duration : float, optional
            The duration of the sweep in seconds (default is 5).
        total_duration : float, optional
            The total duration of the resulting signal including silence (default is 5).
        inverse : bool, optional
            If True, generate the inverse filter of the sweep (default is False).
        amplitude_dbfs : float, optional
            The amplitude of the sweep in dBFS (default is -1).
        start_time : float, optional
            The start time of the sweep within the duration (default is 0).

        Returns
        -------
        ndarray
            The generated logarithmic sweep tone or its inverse.
        """
        linear_amplitude = self.dbfs_to_amplitude(amplitude_dbfs)
        R = np.log(f1 / f0)
        t = np.arange(0, signal_duration, 1.0 / self.sample_rate)
        output = np.sin((2.0 * np.pi * f0 * signal_duration / R) * (np.exp(t * R / signal_duration) - 1)) * linear_amplitude
        if inverse:
            k = np.exp(t * R / signal_duration)
            output = output[::-1] / k

        # Place the generated logarithmic sweep at the specified start time within the total duration
        placed_sweep = self.place_signal(output, start_time, signal_duration, total_duration)
        if self.channel_mode == 'stereo':
            placed_sweep = np.tile(placed_sweep[:, np.newaxis], (1, 2))
        return placed_sweep

    def save_signal(self, signal, file_path, file_name):
        """
        Save the generated signal to a specified path as a WAV file.

        Parameters
        ----------
        signal : ndarray
            The signal to save.
        file_path : str
            The directory path where the signal will be saved.
        file_name : str
            The name of the file.

        Returns
        -------
        str
            The full path to the saved file.
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        full_path = os.path.join(file_path, file_name)
        sf.write(full_path, signal, self.sample_rate, subtype=self.bit_depth)
        print(f"Signal saved to {full_path}")
        return full_path
    