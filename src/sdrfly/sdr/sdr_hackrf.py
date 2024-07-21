import numpy as np
import SoapySDR
import threading
from bluetooth_demod.sdr.sdr_base import SDR

class HackRFSdr(SDR):
    MAX_SAMPLES = 131072

    def __init__(self, center_freq, sample_rate, bandwidth, gain, size):
        super().__init__(center_freq, sample_rate, bandwidth, gain)
        results = SoapySDR.Device.enumerate("driver=hackrf")
        if len(results) == 0:
            raise RuntimeError("No HackRF devices found")
        self.sdr = SoapySDR.Device(results[0])
        self.set_sample_rate(sample_rate)
        self.set_frequency(center_freq)
        self.set_bandwidth(bandwidth)
        self.set_gain(gain)
        self.rx_stream = None
        self.tx_stream = None
        self.running = False
        self.data_lock = threading.Lock()
        self.size = size
        self.sample_buffer = np.zeros(size, dtype=np.complex64)
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
            samples = self.capture_samples(self.size)
            with self.data_lock:
                self.sample_buffer[:] = samples

    def get_latest_samples(self):
        # return self.capture_samples(self.size)  # Increase buffer size to decrease RBW
        with self.data_lock:
            return self.sample_buffer.copy()

    def capture_samples(self, num_samples):
        if self.rx_stream is None:
            self.rx_stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            self.sdr.activateStream(self.rx_stream, SoapySDR.SOAPY_SDR_END_BURST)

        total_samples = np.empty(num_samples, dtype=np.complex64)
        start_idx = 0

        while start_idx < num_samples:
            remaining_samples = num_samples - start_idx
            chunk_samples = min(HackRFSdr.MAX_SAMPLES, remaining_samples)
            samples = np.empty(chunk_samples, dtype=np.complex64)
            sr = self.sdr.readStream(self.rx_stream, [samples], chunk_samples)

            if sr.ret > 0:
                total_samples[start_idx:start_idx + sr.ret] = samples[:sr.ret]
                start_idx += sr.ret

        return total_samples[:start_idx]

    def set_frequency(self, freq):
        self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)

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
