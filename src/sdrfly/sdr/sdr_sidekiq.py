import numpy as np
import SoapySDR
import threading
import matplotlib.pyplot as plt
from sdrfly.sdr.sdr_base import SDR

class SidekiqSdr(SDR):

    def __init__(self, center_freq, sample_rate, bandwidth, gain, size):
        super().__init__(center_freq, sample_rate, bandwidth, gain)
        self.readsize = 1024*1018
        self.sample_buffer = np.zeros(self.readsize, dtype=np.complex64)
        SoapySDR.setLogLevel(SoapySDR.SOAPY_SDR_INFO)
        results = SoapySDR.Device.enumerate("driver=sidekiq")
        if len(results) == 0:
            raise RuntimeError("No SDR devices found")
        self.sdr = SoapySDR.Device(results[0])
        self.set_sample_rate(sample_rate)
        self.set_frequency(center_freq)
        self.set_bandwidth(bandwidth)
        self.set_gain(gain)
        self.rx_stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rx_stream)
        self.tx_stream = None
        self.running = False
        self.data_lock = threading.Lock()
        self.size = size
        self.thread = threading.Thread(target=self._capture_thread)
        self.thread.daemon = True

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _capture_thread(self):
        while self.running:
            self.capture_samples(self.size)

    def get_latest_samples(self):
        return self.sample_buffer.copy()[0:self.size]

    def capture_samples(self, num_samples):
        sr = self.sdr.readStream(self.rx_stream, [self.sample_buffer], self.readsize)
        if sr.ret > 0:
            # print(f"Captured {sr.ret} samples")
            """"""
        else:
            print("Failed to read samples from SDR")
        return self.sample_buffer

    def set_frequency(self, freq):
        try:
            self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
        except Exception as e:
            print(e)
            pass

    def set_sample_rate(self, rate):
        self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, rate)

    def set_bandwidth(self, bandwidth):
        self.sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, bandwidth)

    def set_gain(self, gain):
        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)

    def transmit_samples(self, samples):
        if self.tx_stream is None:
            self.tx_stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32)
            self.sdr.activateStream(self.tx_stream)

        sr = self.sdr.writeStream(self.tx_stream, [samples], len(samples))
        if sr.ret != len(samples):
            print("Transmitted {} samples instead of {}".format(sr.ret, len(samples)))
        else:
            print("Transmitted {} samples".format(len(samples)))

    def stop_transmission(self):
        if self.tx_stream is not None:
            self.sdr.deactivateStream(self.tx_stream)
            self.sdr.closeStream(self.tx_stream)
            self.tx_stream = None

    def close(self):
        if self.rx_stream is not None:
            self.sdr.deactivateStream(self.rx_stream)
            self.sdr.closeStream(self.rx_stream)
            self.rx_stream = None
        if self.tx_stream is not None:
            self.sdr.deactivateStream(self.tx_stream)
            self.sdr.closeStream(self.tx_stream)
            self.tx_stream = None
        self.sdr = None

    def plot_fft(self):
        samples = self.get_latest_samples()
        fft_result = np.fft.fftshift(np.fft.fft(samples))
        fft_magnitude = 20 * np.log10(np.abs(fft_result))
        freq_axis = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/self.sample_rate)) + self.center_freq

        plt.figure(figsize=(10, 6))
        plt.plot(freq_axis / 1e6, fft_magnitude)
        plt.title('FFT of Received Samples')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid()
        plt.show()
