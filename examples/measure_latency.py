from analog_io import HardwareLatencyMeasure

tester = HardwareLatencyMeasure()
tester.select_device_and_channels()
latency_time, latency_index = tester.measure_latency()
