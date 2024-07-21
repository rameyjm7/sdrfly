from abc import ABC, abstractmethod

class SDR(ABC):
    def __init__(self, center_freq, sample_rate, bandwidth, gain):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.gain = gain

    @abstractmethod
    def capture_samples(self, num_samples):
        pass

    @abstractmethod
    def transmit_samples(self, samples):
        pass

    @abstractmethod
    def set_frequency(self, frequency):
        pass

    @abstractmethod
    def close(self):
        pass
