"""Signal Generator Package
Copyright (C) 2024 Francesco Papaleo
"""

from signal_generator import SignalGenerator

gen = SignalGenerator()

# Generate a 440 Hz sine tone
sine_signal = gen.generate_tone(pitch=440, signal_duration=3, total_duration=5, waveform='sin', start_time=1, max_amplitude_dbfs=-3)

# Save the sine tone to a file
gen.save_signal(sine_signal, './', 'sine_tone_440Hz.wav')

square_signal = gen.generate_tone(pitch=440, signal_duration=3, total_duration=5, waveform='sqr', start_time=1, max_amplitude_dbfs=-3)

# Save the square wave to a file
gen.save_signal(square_signal, './', 'square_wave_440Hz.wav')

chirp_signal = gen.generate_chirp(f0=200, f1=600, signal_duration=2, total_duration=5, start_time=0, max_amplitude_dbfs=-3)

# Save the chirp to a file
gen.save_signal(chirp_signal, './', 'chirp_200Hz_to_600Hz.wav')

log_sweep_signal = gen.generate_log_sweep(f0=20, f1=20000, signal_duration=5, total_duration=5, start_time=0, amplitude_dbfs=-3)

# Save the logarithmic sweep to a file
gen.save_signal(log_sweep_signal, './', 'log_sweep_20Hz_to_20kHz.wav')

# Generate and save the inverse of the logarithmic sweep
inverse_log_sweep_signal = gen.generate_log_sweep(f0=20, f1=20000, signal_duration=5, total_duration=5, inverse=True, start_time=0, amplitude_dbfs=-3)
gen.save_signal(inverse_log_sweep_signal, './', 'inverse_log_sweep_20Hz_to_20kHz.wav')
