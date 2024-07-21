from flask import Flask, render_template, jsonify
import numpy as np
import cupy as cp
import threading
import time
from bluetooth_demod.ble_sniffer import BLESniffer
from bluetooth_demod.sdr.sdr_hackrf import HackRFSdr

app = Flask(__name__)

# Parameters
CENTER_FREQ = 2.426e9  # Centered at BLE advertising channel 38 (2426 MHz)
SAMPLE_RATE = 16e6     # 16 MSPS
BANDWIDTH = 16e6       # 16 MHz capture bandwidth
GAIN = 30              # Gain in dB
CAPTURE_DURATION = 1   # Capture duration in seconds
NUM_SAMPLES = int(SAMPLE_RATE * CAPTURE_DURATION)  # Number of samples for 1 second
CHUNK_SIZE = 131072    # Number of samples per chunk from HackRF

# Initialize the BLESniffer
sniffer = BLESniffer(HackRFSdr, CENTER_FREQ, SAMPLE_RATE, BANDWIDTH, GAIN)

data_lock = threading.Lock()
fft_data = []

def capture_data():
    global fft_data
    while True:
        try:
            # Capture samples in chunks and accumulate
            accumulated_samples = cp.empty(NUM_SAMPLES, dtype=cp.complex64)
            start_idx = 0

            while start_idx < NUM_SAMPLES:
                remaining_samples = NUM_SAMPLES - start_idx
                chunk_samples = min(CHUNK_SIZE, remaining_samples)
                samples = sniffer.capture_samples(chunk_samples)

                if len(samples) > 0:
                    accumulated_samples[start_idx:start_idx + len(samples)] = cp.asarray(samples)
                    start_idx += len(samples)
                else:
                    print("Requested {} samples but got {} samples.".format(chunk_samples, len(samples)))

            print("Captured {} samples.".format(NUM_SAMPLES))

            # Compute FFT
            fft_samples = cp.fft.fft(accumulated_samples)
            fft_data = cp.asnumpy(20 * cp.log10(cp.abs(fft_samples)))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error capturing data: {}".format(e))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    with data_lock:
        return jsonify(fft_data.tolist())

if __name__ == '__main__':
    data_thread = threading.Thread(target=capture_data)
    data_thread.start()
    app.run(host="0.0.0.0", port=80, debug=True, use_reloader=False)
