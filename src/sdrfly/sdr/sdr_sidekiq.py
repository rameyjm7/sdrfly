import numpy as np
import SoapySDR
import threading
import matplotlib.pyplot as plt
from sdrfly.sdr.sdr_base import SDR
import time
import os

class SidekiqSdr(SDR):

    def __init__(self, center_freq, sample_rate, bandwidth, gain, size):
        super().__init__(center_freq, sample_rate, bandwidth, gain)
        # Suppress SoapySDR logs by redirecting stderr
        self.devnull = open(os.devnull, 'w')
        self.old_stderr = os.dup(2)  # Duplicate the existing stderr
        os.dup2(self.devnull.fileno(), 2)  # Redirect stderr to /dev/null
        
        SoapySDR.setLogLevel(SoapySDR.SOAPY_SDR_ERROR)  # Set log level to error
        
        self.readsize = 1024 * 1018
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
        self.tx_thread = None  # Thread for async transmission

        # Tracking read statistics
        self.failed_reads = 0
        self.total_reads = 0
        self.last_report_time = time.time()

    def __del__(self):
        os.dup2(self.old_stderr, 2)  # Restore stderr
        self.devnull.close()

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
        self.total_reads += 1  # Increment total reads
        if sr.ret > 0:
            pass  # Successfully captured samples
        else:
            self.failed_reads += 1  # Increment failed reads

        # Print failure ratio once a minute
        current_time = time.time()
        if current_time - self.last_report_time > 60:  # 60 seconds interval
            failure_ratio = self.failed_reads / self.total_reads
            print(f"Read failures: {self.failed_reads}/{self.total_reads} ({failure_ratio:.2%})")
            self.last_report_time = current_time  # Reset report time

        return self.sample_buffer

    def set_frequency(self, freq):
        try:
            self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
        except Exception as e:
            print(e)

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
        freq_axis = np.fft.fftshift(np.fft.fftfreq(len(samples), 1 / self.sample_rate)) + self.center_freq

        plt.figure(figsize=(10, 6))
        plt.plot(freq_axis / 1e6, fft_magnitude)
        plt.title('FFT of Received Samples')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid()
        plt.show()

    def transmit_data_async(self, samples, duration=1):
        """
        Transmit data asynchronously in a separate thread.

        Parameters:
            samples (np.array): The samples to transmit.
            duration (float): Duration of the transmission in seconds.
        """
        if self.tx_thread is not None and self.tx_thread.is_alive():
            print("A transmission is already in progress. Please wait for it to finish.")
            return 1

        def transmit():
            try:
                num_repeats = int(self.sample_rate * duration / len(samples))
                repeated_samples = np.tile(samples, num_repeats)
                print(f"Transmitting {len(repeated_samples)} samples asynchronously for {duration} seconds...")
                self.transmit_samples(repeated_samples)
                time.sleep(duration)
                self.stop_transmission()
                print("Asynchronous transmission complete.")
            except Exception as e:
                print(f"Error during asynchronous transmission: {e}")

        self.tx_thread = threading.Thread(target=transmit)
        self.tx_thread.daemon = True
        self.tx_thread.start()
