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


def dbfs_to_linear(dbfs: float) -> float:
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

    def gen_sin(
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
        amplitude = dbfs_to_linear(amplitude_dbfs)
        return amplitude * np.sin(2 * np.pi * frequency * t)

    def gen_saw(
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
        amplitude = dbfs_to_linear(amplitude_dbfs)
        return amplitude * scipy.signal.sawtooth(2 * np.pi * frequency * t)

    def gen_sqr(
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
        amplitude = dbfs_to_linear(amplitude_dbfs)
        return amplitude * scipy.signal.square(2 * np.pi * frequency * t)

    def gen_pulse(
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
        amplitude = dbfs_to_linear(amplitude_dbfs)
        pulse = np.zeros_like(t)
        pulse_samples = int(self.sample_rate * pulse_width)
        pulse[:pulse_samples] = amplitude  # Set the pulse for the beginning
        return pulse

    def adjust_channels(self, signal: np.ndarray) -> np.ndarray:
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
        waveform_method = getattr(self, f"gen_{waveform}")
        signal = waveform_method(frequency, duration, amplitude_dbfs)
        signal = self.adjust_channels(signal)
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
        test_pulse_signal = self.signal_generator.gen_pulse(
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


class AnalogProcessor:
    """
    A class used to process audio files with analog effects.

    Parameters
    ----------
    input_folder : str, optional
        The input folder containing the dry audio files (default is "dry").
    output_folder : str, optional
        The output folder for the wet audio files (default is "wet").
    sample_rate : int, optional
        The sample rate to be used (default is 48000).
    device_index : int, optional
        The index of the audio device to be used (default is 6).
    input_channel_index : int, optional
        The index of the input channel to be used (default is 1).
    output_channel_index : int, optional
        The index of the output channel to be used (default is 3).
    latency_samples : int, optional
        The latency in samples for the audio device (default is 9526).
    wait_time : float, optional
        The time to wait after processing each file (default is 4.0).
    """

    def __init__(
        self,
        input_folder: str = "dry",
        output_folder: str = "wet",
        sample_rate: int = 48000,
        device_index: int = 6,
        input_channel_index: int = 1,
        output_channel_index: int = 3,
        latency_samples: int = 9526,
        wait_time: float = 4.0,
    ) -> None:
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.input_channel_index = input_channel_index
        self.output_channel_index = output_channel_index
        self.latency_samples = latency_samples
        self.wait_time = wait_time

        # Ensure output directory exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    @staticmethod
    def adjust_recording_for_latency(
        recorded_signal: np.ndarray, latency_in_samples: int
    ) -> np.ndarray:
        """
        Adjust the recording to remove the specified latency and compensate for the end.
        """
        # Remove latency at the beginning
        adjusted_signal = recorded_signal[latency_in_samples:]

        # Compensate for the removed latency by adding silence at the end
        silence_duration_samples = latency_in_samples
        silence = np.zeros(silence_duration_samples)
        compensated_signal = np.concatenate([adjusted_signal, silence])

        return compensated_signal

    def process_files(self) -> None:
        """
        Process all WAV files in the input folder, applying effects and saving to the output folder.
        """
        file_paths = [
            os.path.join(self.input_folder, f)
            for f in os.listdir(self.input_folder)
            if f.endswith(".wav")
        ]

        for file_path in file_paths:
            audio, fs = sf.read(file_path)

            # Ensure the audio is mono for playback
            if audio.ndim > 1:
                audio = audio[:, 0]  # Use the first channel if stereo

            # Play and record simultaneously
            recorded_signal = sd.playrec(
                audio[:, np.newaxis],
                samplerate=fs,
                input_mapping=[self.input_channel_index],
                output_mapping=[self.output_channel_index],
                device=self.device_index,
                channels=1,
            )
            sd.wait()  # Wait until recording is finished

            # Adjust recording for latency and compensate at the end
            adjusted_recording = self.adjust_recording_for_latency(
                recorded_signal[:, 0], self.latency_samples
            )

            # Modify the filename pattern from 'dry_' to 'wet_'
            new_filename = os.path.basename(file_path).replace("dry_", "wet_")
            output_file_path = os.path.join(self.output_folder, new_filename)

            # Save the adjusted recording
            sf.write(output_file_path, adjusted_recording, fs, subtype="PCM_24")

            # Wait for the external device to become silent
            time.sleep(self.wait_time)

        print("Finished processing all files.")
