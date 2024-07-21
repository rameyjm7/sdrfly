import numpy as np
from scipy.fftpack import fft, ifft
from concurrent.futures import ThreadPoolExecutor
from bluetooth_demod.channelizers.channelizer_base import ChannelizerBase

class ChannelizerFFT(ChannelizerBase):
    def __init__(self, num_channels=10, channel_bw=1e6, sample_rate=10e6):
        super().__init__(num_channels, channel_bw, sample_rate)

    def channelize(self, samples):
        return self._channelize_fft_parallel(samples, self.num_channels, self.channel_bw, self.sample_rate)

    def _channelize_fft_parallel(self, samples, num_channels, channel_bw, sample_rate):
        num_samples = len(samples)
        fft_size = int(sample_rate // channel_bw)
        channel_samples = np.zeros((num_channels, num_samples // fft_size), dtype=np.complex64)
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_channel_fft, samples, fft_size, i) for i in range(num_channels)]
            for i, future in enumerate(futures):
                channel_samples[i] = future.result()
        
        return channel_samples

    def _process_channel_fft(self, samples, fft_size, channel):
        num_samples = len(samples)
        freq_bins = fft(samples)
        start_bin = channel * fft_size
        end_bin = start_bin + fft_size
        channel_freq_bins = np.zeros_like(freq_bins)
        channel_freq_bins[start_bin:end_bin] = freq_bins[start_bin:end_bin]
        return ifft(channel_freq_bins)[:num_samples // fft_size]
