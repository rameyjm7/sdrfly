import numpy as np
import numba
from numba import njit, prange

class ChannelizerNumba:
    def __init__(self, num_channels, channel_bw, sample_rate):
        self.num_channels = num_channels
        self.channel_bw = channel_bw
        self.sample_rate = sample_rate
        self.decimation_factor = int(sample_rate / channel_bw)
        self.polyphase_filter = self.create_polyphase_filter(num_channels, self.decimation_factor)

    def create_polyphase_filter(self, num_channels, decimation_factor):
        # Create a low-pass filter
        num_taps = 128
        cutoff = 1 / num_channels
        taps = np.sinc(2 * cutoff * (np.arange(num_taps) - (num_taps - 1) / 2))
        taps *= np.hamming(num_taps)
        taps /= np.sum(taps)

        # Adjust the number of taps to be divisible by the decimation factor
        num_taps = (len(taps) // decimation_factor) * decimation_factor
        taps = taps[:num_taps]

        # Create polyphase components
        polyphase_filter = np.zeros((num_channels, decimation_factor, num_taps // decimation_factor))
        for i in range(num_channels):
            for j in range(decimation_factor):
                polyphase_filter[i, j, :] = taps[j::decimation_factor]

        return polyphase_filter

    def channelize(self, samples):
        return polyphase_channelizer(samples, self.polyphase_filter, self.num_channels, self.decimation_factor)

# Numba JIT function for efficiency
@njit(parallel=True)
def polyphase_channelizer(samples, polyphase_filter, num_channels, decimation_factor):
    num_samples = len(samples)
    num_output_samples = num_samples // num_channels
    channel_samples = np.zeros((num_channels, num_output_samples), dtype=np.complex64)
    for i in prange(num_channels):
        filtered_samples = np.convolve(samples, polyphase_filter[i].flatten(), mode='valid')
        channel_samples[i, :] = filtered_samples[:num_output_samples]
    return channel_samples
