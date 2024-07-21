import numpy as np
from sdrfly.sdr.sdr_base import SDR

class SimulatedBluetoothSDR(SDR):
    def __init__(self, center_freq, sample_rate, bandwidth, gain):
        super().__init__(center_freq, sample_rate, bandwidth, gain)

    def capture_samples(self, num_samples):
        t = np.arange(num_samples) / self.sample_rate
        freq = 1e6  # 1 MHz Bluetooth-like signal
        signal = np.exp(2j * np.pi * freq * t)
        noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        samples = (signal + noise).astype(np.complex64)  # Ensure the samples are complex64

        # Adding a simulated peak
        peak_position = num_samples // 2
        samples[peak_position:peak_position + 10] += 2.0
        return samples

    def set_frequency(self, frequency):
        self.center_freq = frequency

    def transmit_samples(self, samples):
        # Simulate transmission by printing a message
        print(f"Simulated transmission of {len(samples)} samples")

    def close(self):
        pass  # No resources to release in simulation
