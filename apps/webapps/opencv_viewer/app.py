from flask import Flask, render_template, Response
import numpy as np
import cv2
from datetime import datetime
import os
from multiprocessing import Process, Queue
import pickle
from bluetooth_demod.sdr.sdr_hackrf import HackRFSdr

app = Flask(__name__)

# Set the Qt platform to offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# HackRF setup using HackRFSdr
center_freq = 2.4e9  # Center frequency for Bluetooth
sample_rate = 10e6  # Sample rate
bandwidth = 10e6  # Bandwidth
gain = 20  # Gain

hackrf_sdr = HackRFSdr(center_freq=center_freq, sample_rate=sample_rate, bandwidth=bandwidth, gain=gain)
sample_buffer = np.zeros(1024, dtype=np.complex64)  # Initial buffer

def hackrf_callback():
    global sample_buffer
    sample_buffer = hackrf_sdr.capture_samples(1024)

def generate_fft_image(q):
    from PySide6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene
    from PySide6.QtGui import QImage, QPainter, QPixmap
    from PySide6.QtCore import QBuffer, QIODevice
    import pyqtgraph as pg

    # Create a single instance of QApplication
    qt_app = QApplication([])

    while True:
        sample_buffer = q.get()
        
        # Perform FFT
        fft_result = np.fft.fftshift(np.fft.fft(sample_buffer))
        fft_magnitude = 20 * np.log10(np.abs(fft_result))

        # Generate the FFT plot using pyqtgraph
        plt = pg.PlotWidget()
        plt.plot(np.linspace(-sample_rate/2, sample_rate/2, len(fft_magnitude)), fft_magnitude, pen='y')
        plt.setTitle("FFT of Signal")
        plt.setLabel('left', 'Magnitude (dB)')
        plt.setLabel('bottom', 'Frequency')

        # Render plot to an image using QGraphicsView and QGraphicsScene
        scene = QGraphicsScene()
        scene.addWidget(plt)
        view = QGraphicsView(scene)
        view.resize(640, 480)
        pixmap = QPixmap(view.size())
        painter = QPainter(pixmap)
        view.render(painter)
        painter.end()

        buffer = QBuffer()
        buffer.open(QIODevice.ReadWrite)
        pixmap.save(buffer, "PNG")
        buffer.seek(0)
        img = buffer.data().data()

        q.put(img)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    q = Queue()
    p = Process(target=generate_fft_image, args=(q,))
    p.start()

    while True:
        # Capture new samples from HackRF
        hackrf_callback()
        q.put(sample_buffer)

        # Get the FFT image from the separate process
        fft_img_data = q.get()
        fft_img = np.frombuffer(fft_img_data, np.uint8)
        fft_img = cv2.imdecode(fft_img, cv2.IMREAD_COLOR)

        # Create an image with black background
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Get the current time
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Put the current time text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, current_time, (50, 450), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Overlay the FFT image on the frame
        fft_height, fft_width, _ = fft_img.shape
        frame[0:fft_height, 0:fft_width] = fft_img

        # Encode the image as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def run_flask():
    app.run(host='0.0.0.0', port=80, threaded=True)

if __name__ == '__main__':
    run_flask()
