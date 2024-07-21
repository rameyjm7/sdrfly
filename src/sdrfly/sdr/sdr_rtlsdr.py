import SoapySDR
import numpy as np
import time
from sdrfly.sdr.sdr_base import SDR

class RTLSDR(SDR):
    def __init__(self, center_freq, sample_rate, bandwidth, gain):
        super().__init__(center_freq, sample_rate, bandwidth, gain)

        # Find and open the RTL-SDR device
        results = SoapySDR.Device.enumerate("driver=rtlsdr")
        if len(results) == 0:
            raise RuntimeError("No RTL-SDR devices found")

        self.sdr = SoapySDR.Device(results[0])
        self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, sample_rate)
        self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_freq)
        self.sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, bandwidth)
        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
        self.rx_stream = None

    def capture_samples(self, num_samples):
        if self.rx_stream is None:
            self.rx_stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            self.sdr.activateStream(self.rx_stream)
            time.sleep(0.1)  # Small delay to allow stream to activate

        samples = np.empty(num_samples, dtype=np.complex64)
        sr = self.sdr.readStream(self.rx_stream, [samples], num_samples)

        # print("readStream returned: {} samples, flags: {}, timeNs: {}".format(sr.ret, sr.flags, sr.timeNs))

        if sr.ret <= 0:
            print("Requested {} samples but got {} samples.".format(num_samples, sr.ret))
            return np.array([], dtype=np.complex64)

        return samples[:sr.ret]

    def transmit_samples(self, samples):
        raise NotImplementedError("RTL-SDR does not support transmission")

    def set_frequency(self, freq):
        self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
        print("Frequency set to {} MHz".format(freq / 1e6))

    def close(self):
        if self.rx_stream is not None:
            self.sdr.deactivateStream(self.rx_stream)
            self.sdr.closeStream(self.rx_stream)
            self.rx_stream = None
        self.sdr = None
        print("RTL-SDR closed")
