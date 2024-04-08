from analog_io import HardwareLatencyMeasure, SignalGenerator, HardwareAnalogDevice

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# tester = HardwareLatencyMeasure()
# tester.select_device_and_channels()
# latency_time, latency_index = tester.measure_latency()

sg = SignalGenerator(sample_rate=48000, bit_depth="PCM_24", channel_mode="mono")
log_sweep = sg.log_sweep(20.0, 20000.0, 10.0, -1.0, inverse=False).reshape(-1, 1)

inv_filter = sg.log_sweep(20.0, 20000.0, 10.0, -1.0, inverse=True).reshape(-1, 1)

print(log_sweep.shape, inv_filter.shape)

# Initialize HardwareAnalogDevice 
device = HardwareAnalogDevice(sample_rate=48000, device_index=6, # Set your device index
                              input_channel_index=1, # Set your input channel index
                              output_channel_index=3, # Set your output channel index
                              latency_samples=9526, wait_time=4.0)

# Play and record the log sweep through the external analog device
recorded_sweep = device.process_audio(log_sweep).reshape(-1, 1)

# Convolve the recorded signal with the inverse filter to get the impulse response
impulse_response = scipy.signal.fftconvolve(recorded_sweep, inv_filter, mode='full')

# Assuming the impulse response is centered, trim it to the original log_sweep length
impulse_response = impulse_response[len(log_sweep):]

# Plot the impulse response
plt.figure(figsize=(10, 6))
plt.plot(impulse_response)
plt.title("Impulse Response of the External Analog Device")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
