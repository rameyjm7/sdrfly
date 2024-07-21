import numpy as np
import cupy as cp

class ChannelizerCuPy:
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
        samples_gpu = cp.asarray(samples)
        polyphase_filter_gpu = cp.asarray(self.polyphase_filter)
        return polyphase_channelizer(samples_gpu, polyphase_filter_gpu, self.num_channels, self.decimation_factor)

def polyphase_channelizer(samples, polyphase_filter, num_channels, decimation_factor):
    num_samples = len(samples)
    num_output_samples = num_samples // decimation_factor
    channel_samples = cp.zeros((num_channels, num_output_samples), dtype=cp.complex64)

    for i in range(num_channels):
        filtered_samples = cp.zeros(num_output_samples, dtype=cp.complex64)
        for j in range(decimation_factor):
            conv_result = cp.convolve(samples[j::decimation_factor], polyphase_filter[i, j], mode='valid')
            # Ensure the shapes match by zero-padding or slicing
            if len(conv_result) < num_output_samples:
                conv_result = cp.pad(conv_result, (0, num_output_samples - len(conv_result)), 'constant')
            else:
                conv_result = conv_result[:num_output_samples]
            filtered_samples += conv_result
        channel_samples[i, :] = filtered_samples

    return channel_samples
