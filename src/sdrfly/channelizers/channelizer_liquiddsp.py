import numpy as np
import ctypes
from sdrfly.channelizers.channelizer_base import ChannelizerBase

# Load the LiquidDSP library
libliquid = ctypes.CDLL('/usr/local/lib/libliquid.so')

class ChannelizerLiquidDSP(ChannelizerBase):
    def __init__(self, num_channels=10, channel_bw=1e6, sample_rate=10e6):
        super().__init__(num_channels, channel_bw, sample_rate)
        
        self.nco_crcf_create = libliquid.nco_crcf_create
        self.nco_crcf_create.restype = ctypes.c_void_p
        
        self.nco_crcf_destroy = libliquid.nco_crcf_destroy
        self.nco_crcf_destroy.argtypes = [ctypes.c_void_p]
        
        self.nco_crcf_set_frequency = libliquid.nco_crcf_set_frequency
        self.nco_crcf_set_frequency.argtypes = [ctypes.c_void_p, ctypes.c_float]
        
        self.nco_crcf_mix_down = libliquid.nco_crcf_mix_down
        self.nco_crcf_mix_down.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_complex), ctypes.POINTER(ctypes.c_complex), ctypes.c_int]
        
        self.firfilt_crcf_create_kaiser = libliquid.firfilt_crcf_create_kaiser
        self.firfilt_crcf_create_kaiser.restype = ctypes.c_void_p
        self.firfilt_crcf_create_kaiser.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.c_float]
        
        self.firfilt_crcf_destroy = libliquid.firfilt_crcf_destroy
        self.firfilt_crcf_destroy.argtypes = [ctypes.c_void_p]
        
        self.firfilt_crcf_execute_block = libliquid.firfilt_crcf_execute_block
        self.firfilt_crcf_execute_block.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        
        self.fir_filter = self._create_fir_filter()

    def _create_fir_filter(self):
        num_taps = 128
        cutoff_freq = self.channel_bw / (2 * self.sample_rate)
        taps = np.zeros(num_taps, dtype=np.float32)
        return self.firfilt_crcf_create_kaiser(taps.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), num_taps, 60.0, cutoff_freq)

    def _channelize_liquiddsp(self, samples, num_channels, channel_bw, sample_rate):
        num_samples = len(samples)
        channel_samples = np.zeros((num_channels, num_samples), dtype=np.complex64)
        
        for i in range(num_channels):
            nco = self.nco_crcf_create()
            freq_shift = (i - num_channels // 2) * channel_bw
            self.nco_crcf_set_frequency(nco, ctypes.c_float(2 * np.pi * freq_shift / sample_rate))
            mixed_down_samples = self.mix_down_liquiddsp(samples, freq_shift)
            filtered_samples = self.apply_fir_filter_liquiddsp(mixed_down_samples)
            channel_samples[i] = filtered_samples
            self.nco_crcf_destroy(nco)
        
        return channel_samples

    def mix_down_liquiddsp(self, samples, freq_shift):
        num_samples = len(samples)
        mixed_down_samples = np.zeros(num_samples, dtype=np.complex64)
        nco = self.nco_crcf_create()
        self.nco_crcf_set_frequency(nco, freq_shift)
        self.nco_crcf_mix_down(nco, samples.ctypes.data_as(ctypes.POINTER(ctypes.c_complex)), mixed_down_samples.ctypes.data_as(ctypes.POINTER(ctypes.c_complex)), num_samples)
        self.nco_crcf_destroy(nco)
        return mixed_down_samples

    def apply_fir_filter_liquiddsp(self, samples):
        num_samples = len(samples)
        filtered_samples = np.zeros(num_samples, dtype=np.complex64)
        fir_filter = self.fir_filter
        samples_real_imag = np.vstack((samples.real, samples.imag)).reshape((-1,), order='F').astype(np.float32)
        filtered_samples_real_imag = np.zeros(samples_real_imag.shape, dtype=np.float32)
        self.firfilt_crcf_execute_block(fir_filter, samples_real_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), filtered_samples_real_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), num_samples)
        filtered_samples = filtered_samples_real_imag.view(np.complex64)
        self.firfilt_crcf_destroy(fir_filter)
        return filtered_samples

    def channelize(self, samples):
        return self._channelize_liquiddsp(samples, self.num_channels, self.channel_bw, self.sample_rate)
