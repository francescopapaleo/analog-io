"""Analog I/O Tools for Python
Copyright (C) 2024 Francesco Papaleo
"""

import os
import time
import numpy as np
import scipy.signal
import sounddevice as sd
import soundfile as sf
from typing import Tuple, List


def dbfs_to_amp(dbfs: float) -> float:
    """
    Convert dBFS to a linear amplitude scale.

    Parameters
    ----------
    dbfs : float
        The dBFS value to convert.

    Returns
    -------
    float
        The linear amplitude scale value.
    """
    return 10 ** (dbfs / 20)


def amp_to_dbfs(amplitude: float) -> float:
    """
    Convert a linear amplitude scale to dBFS.

    Parameters
    ----------
    amplitude : float
        The linear amplitude scale value to convert.

    Returns
    -------
    float
        The dBFS value.
    """
    return 20 * np.log10(amplitude)


class SignalGenerator:
    def __init__(
        self,
        sample_rate: int = 48000,
        bit_depth: str = "PCM_24",
        channel_mode: str = "mono",
    ):
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.channel_mode = channel_mode

    def generate_time_array(self, duration: float) -> np.ndarray:
        """
        Generate a time array for the given duration.

        Parameters
        ----------
        duration : float
            The duration for which to generate the time array.

        Returns
        -------
        np.ndarray
            The generated time array.
        """
        return np.linspace(
            0, duration, int(self.sample_rate * duration), endpoint=False
        )

    def sin(
        self, frequency: float, duration: float, amplitude_dbfs: float
    ) -> np.ndarray:
        """
        Generate a sine waveform.

        Parameters
        ----------
        frequency : float
            The frequency of the sine waveform.
        duration : float
            The duration of the sine waveform.
        amplitude_dbfs : float
            The amplitude of the sine waveform in dBFS.

        Returns
        -------
        np.ndarray
            The generated sine waveform.
        """
        t = self.generate_time_array(duration)
        amplitude = dbfs_to_amp(amplitude_dbfs)
        return amplitude * np.sin(2 * np.pi * frequency * t)

    def saw(
        self, frequency: float, duration: float, amplitude_dbfs: float
    ) -> np.ndarray:
        """
        Generate a sawtooth waveform.

        Parameters
        ----------
        frequency : float
            The frequency of the sawtooth waveform.
        duration : float
            The duration of the sawtooth waveform.
        amplitude_dbfs : float
            The amplitude of the sawtooth waveform in dBFS.

        Returns
        -------
        np.ndarray
            The generated sawtooth waveform.
        """
        t = self.generate_time_array(duration)
        amplitude = dbfs_to_amp(amplitude_dbfs)
        return amplitude * scipy.signal.sawtooth(2 * np.pi * frequency * t)

    def sqr(
        self, frequency: float, duration: float, amplitude_dbfs: float
    ) -> np.ndarray:
        """
        Generate a square waveform.

        Parameters
        ----------
        frequency : float
            The frequency of the square waveform.
        duration : float
            The duration of the square waveform.
        amplitude_dbfs : float
            The amplitude of the square waveform in dBFS.

        Returns
        -------
        np.ndarray
            The generated square waveform.
        """
        t = self.generate_time_array(duration)
        amplitude = dbfs_to_amp(amplitude_dbfs)
        return amplitude * scipy.signal.square(2 * np.pi * frequency * t)

    def pulse(
        self, duration: float, amplitude_dbfs: float, pulse_width: float
    ) -> np.ndarray:
        """
        Generate a pulse waveform.

        Parameters
        ----------
        duration : float
            The duration of the pulse waveform.
        amplitude_dbfs : float
            The amplitude of the pulse waveform in dBFS.
        pulse_width : float
            The width of the pulse waveform.

        Returns
        -------
        np.ndarray
            The generated pulse waveform.
        """
        t = self.generate_time_array(duration)
        amplitude = dbfs_to_amp(amplitude_dbfs)
        pulse = np.zeros_like(t)
        pulse_samples = int(self.sample_rate * pulse_width)
        pulse[:pulse_samples] = amplitude  # Set the pulse for the beginning
        return pulse

    def log_sweep(
        self,
        f0: float,
        f1: float,
        duration: float,
        amplitude_dbfs: float,
        inverse: bool = False,
    ) -> np.ndarray:
        """
        Generate a logarithmic swept sine waveform.

        Parameters
        ----------
        f0 : float
            The start frequency of the sweep.
        f1 : float
            The end frequency of the sweep.
        duration : float
            The duration of the sweep.
        amplitude_dbfs : float
            The amplitude of the sweep in dBFS.
        inverse : bool, optional
            Whether to use an inverse sweep (default is False).

        Returns
        -------
        np.ndarray
            The generated swept sine waveform.
        """
        t = self.generate_time_array(duration)
        amplitude = dbfs_to_amp(amplitude_dbfs)
        R = np.log(f1 / f0)
        output = np.sin(
            (2.0 * np.pi * f0 * duration / R) * (np.exp(t * R / duration) - 1)
        )
        if inverse:
            k = np.exp(t * R / duration)
            output = output[::-1] / k
        return amplitude * output

    def mono_to_stereo(self, signal: np.ndarray) -> np.ndarray:
        """
        Adjust the signal for the specified channel mode.

        Parameters
        ----------
        signal : np.ndarray
            The input signal.

        Returns
        -------
        np.ndarray
            The adjusted signal.
        """
        if self.channel_mode == "mono":
            return signal
        elif self.channel_mode == "stereo":
            return np.stack((signal, signal), axis=-1)

    def place_signal(
        self, signal: np.ndarray, start_time: float, total_duration: float
    ) -> np.ndarray:
        """
        Place the signal at start_time within a duration of silence.

        Parameters
        ----------
        signal : np.ndarray
            The signal to be placed.
        start_time : float
            The start time for the signal.
        total_duration : float
            The total duration for the signal.

        Returns
        -------
        np.ndarray
            The placed signal.
        """
        if start_time < 0 or total_duration <= 0:
            raise ValueError("start_time and total_duration must be positive values.")

        total_samples = int(self.sample_rate * total_duration)
        start_idx = int(self.sample_rate * start_time)
        if start_idx < 0 or start_idx >= total_samples:
            raise ValueError(
                "start_time is out of the allowed range for the given total_duration."
            )

        silence_before = np.zeros(start_idx)
        silence_after = np.zeros(total_samples - len(signal) - len(silence_before))
        if len(silence_after) < 0:
            raise ValueError("The signal duration exceeds the total_duration.")
        placed_signal = np.concatenate([silence_before, signal, silence_after])
        return placed_signal

    def generate_tone(
        self,
        waveform: str,
        frequency: float,
        duration: float,
        amplitude_dbfs: float,
        start_time: float,
        total_duration: float,
    ) -> np.ndarray:
        """
        Dynamically generate a tone based on the specified waveform.

        Parameters
        ----------
        waveform : str
            The waveform to be generated (sin, saw, sqr, pulse).
        frequency : float
            The frequency of the waveform.
        duration : float
            The duration of the waveform.
        amplitude_dbfs : float
            The amplitude of the waveform in dBFS.
        start_time : float
            The start time for the waveform.
        total_duration : float
            The total duration for the waveform.

        Returns
        -------
        np.ndarray
            The generated waveform.
        """
        waveform_method = getattr(self, f"{waveform}")
        signal = waveform_method(frequency, duration, amplitude_dbfs)
        signal = self.mono_to_stereo(signal)
        return self.place_signal(signal, start_time, total_duration)


class HardwareLatencyMeasure:
    """
    A class used to measure the latency of an audio device.

    Parameters
    ----------
    device_index : int
        The index of the audio device to be tested.
    input_channel_index : int
        The index of the input channel to be used.
    output_channel_index : int
        The index of the output channel to be used.
    sample_rate : int, optional
        The sample rate to be used (default is 48000).
    duration : int, optional
        The duration of the signal in seconds (default is 5).
    pulse_width : float, optional
        The width of the pulse in seconds (default is 0.001).
    pulse_amplitude_dbfs : int, optional
        The amplitude of the pulse in dBFS (default is -1).
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        duration: int = 5,
        pulse_width: float = 0.001,
        amplitude_dbfs: int = -1,
    ) -> None:
        self.signal_generator = SignalGenerator(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.duration = duration
        self.pulse_width = pulse_width
        self.amplitude_dbfs = amplitude_dbfs
        self.start_time = duration / 2
        self.device_index = None
        self.input_channel_index = None
        self.output_channel_index = None

    @staticmethod
    def query_devices() -> List[dict]:
        """Prints available audio devices and returns the device list."""
        devices = sd.query_devices()
        print("Available audio devices:")
        for index, device in enumerate(devices):
            print(f"{index}: {device['name']}")
        return devices

    def select_device_and_channels(self) -> None:
        """Allows the user to select the audio device and channel indices."""
        devices = self.query_devices()
        self.device_index = int(input("Enter the index of the desired audio device: "))
        self.input_channel_index = int(input("Enter the input channel index: "))
        self.output_channel_index = int(input("Enter the output channel index: "))

    def generate_pulse_signal(self) -> np.ndarray:
        """
        Generate a pulse signal for testing.

        Returns
        -------
        numpy.ndarray
            The generated pulse signal. Shape: (duration * sample_rate, 1)
        """
        test_pulse_signal = self.signal_generator.pulse(
            self.duration, self.amplitude_dbfs, self.pulse_width
        )
        return test_pulse_signal.reshape(-1, 1)

    def find_delay(
        self, original: np.ndarray, recorded: np.ndarray
    ) -> Tuple[float, int]:
        """
        Find the delay between the original and recorded signals.

        Parameters
        ----------
        original : numpy.ndarray
            The original signal.
        recorded : numpy.ndarray
            The recorded signal.

        Returns
        -------
        float
            The delay time in seconds.
        int
            The delay index.
        """
        correlation = np.correlate(recorded, original, mode="full")
        delay_index = np.argmax(correlation) - len(original) + 1
        delay_time = delay_index / self.sample_rate
        return delay_time, delay_index

    def measure_latency(self) -> tuple[float, int]:
        """
        Measure the latency of the audio device.

        Returns
        -------
        float
            The latency time in seconds.
        int
            The latency index.
        """
        playback_signal = self.generate_pulse_signal()
        recorded_signal = sd.playrec(
            playback_signal,
            samplerate=self.sample_rate,
            input_mapping=[self.input_channel_index],
            output_mapping=[self.output_channel_index],
            device=self.device_index,
            channels=1,
        )
        sd.wait()  # Wait until recording is finished

        recorded_mono = recorded_signal[:, 0]
        latency_time, latency_index = self.find_delay(
            playback_signal[:, 0], recorded_mono
        )
        print(f"Estimated latency: {latency_time:.3f} seconds, {latency_index} samples")
        return latency_time, latency_index


class HardwareAnalogDevice:
    def __init__(
        self,
        sample_rate: int = 48000,
        device_index: int = None,
        input_channel_index: int = None,
        output_channel_index: int = None,
        latency_samples: int = 0,
        wait_time: float = 4.0,
    ):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.input_channel_index = input_channel_index
        self.output_channel_index = output_channel_index
        self.latency_samples = latency_samples
        self.wait_time = wait_time

    def adjust_recording_for_latency(self, recorded_signal: np.ndarray) -> np.ndarray:
        """
        Adjust the recording to remove the specified latency and compensate for the end.
        """
        # Remove latency at the beginning
        adjusted_signal = recorded_signal[self.latency_samples :]

        # Compensate for the removed latency by adding silence at the end
        silence = np.zeros(self.latency_samples)
        compensated_signal = np.concatenate([adjusted_signal, silence])

        return compensated_signal

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Process the given audio array through an external device and adjust for latency.

        Parameters
        ----------
        audio : np.ndarray
            The audio array to be processed.

        Returns
        -------
        np.ndarray
            The processed audio array.
        """
        if (
            self.device_index is None
            or self.input_channel_index is None
            or self.output_channel_index is None
        ):
            raise ValueError(
                "Device and channel indices must be set before processing."
            )

        # Ensure the audio is mono for playback
        if audio.ndim > 1:
            audio = audio[:, 0]  # Use the first channel if stereo

        # Play and record simultaneously
        recorded_signal = sd.playrec(
            audio[:, np.newaxis],
            samplerate=self.sample_rate,
            input_mapping=[self.input_channel_index],
            output_mapping=[self.output_channel_index],
            device=self.device_index,
            channels=1,
        )
        sd.wait()  # Wait until recording is finished

        # Adjust recording for latency and compensate at the end
        adjusted_recording = self.adjust_recording_for_latency(recorded_signal[:, 0])

        # Wait for the external device to become silent
        time.sleep(self.wait_time)

        return adjusted_recording
