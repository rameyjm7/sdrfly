import SoapySDR
import numpy as np
import logging
import time
from sdrfly.sdr.sdr_base import SDR

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class AirspySDR(SDR):
    def __init__(self, center_freq, sample_rate, bandwidth, gain):
        super().__init__(center_freq, sample_rate, bandwidth, gain)

        # Find and open the Airspy device
        results = SoapySDR.Device.enumerate("driver=airspy")
        if len(results) == 0:
            raise RuntimeError("No Airspy devices found")

        self.sdr = SoapySDR.Device(results[0])
        self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, sample_rate)
        self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_freq)
        self.sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, bandwidth)
        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
        self.rx_stream = None
        self.tx_stream = None

    def capture_samples(self, num_samples):
        if self.rx_stream is None:
            self.rx_stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            self.sdr.activateStream(self.rx_stream)
            time.sleep(0.1)  # Small delay to allow stream to activate

        samples = np.empty(num_samples, dtype=np.complex64)
        sr = self.sdr.readStream(self.rx_stream, [samples], num_samples)

        logger.debug(f"readStream returned: {sr.ret} samples, flags: {sr.flags}, timeNs: {sr.timeNs}")

        if sr.ret <= 0:
            logger.warning(f"Requested {num_samples} samples but got {sr.ret} samples.")
            return np.array([], dtype=np.complex64)

        return samples[:sr.ret]

    def transmit_samples(self, samples):
        if self.tx_stream is None:
            self.tx_stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32)
            self.sdr.activateStream(self.tx_stream)

        samples = samples.astype(np.complex64)
        self.sdr.writeStream(self.tx_stream, [samples], len(samples))
        logger.info(f"Transmitted {len(samples)} samples")

    def stop_transmission(self):
        if self.tx_stream is not None:
            self.tune_away()
            self.sdr.deactivateStream(self.tx_stream)
            self.sdr.closeStream(self.tx_stream)
            self.tx_stream = None
            logger.info("Stopped transmission and closed TX stream")

    def tune_away(self):
        # Tune to a frequency far away from the passband of the RX radio
        self.set_frequency(100e6)  # Example: Tune to 100 MHz
        logger.info("Tuned TX radio away from the passband")

    def set_frequency(self, freq):
        self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
        self.sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, freq)
        logger.info(f"Frequency set to {freq / 1e6} MHz")

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
        logger.info("Airspy SDR closed")
