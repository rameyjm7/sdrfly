import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import spectrogram

def plot_fft(samples, sample_rate, title):
    fft_samples = fft(samples)
    freqs = np.fft.fftfreq(len(fft_samples), 1 / sample_rate)
    plt.figure(figsize=(10, 6))
    plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(20 * np.log10(np.abs(fft_samples))))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.show()

def plot_fft_with_peak(samples, sample_rate, title, peak_freqs=None):
    fft_samples = fft(samples)
    freqs = np.fft.fftfreq(len(fft_samples), 1 / sample_rate)
    magnitude = 20 * np.log10(np.abs(fft_samples))
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(magnitude))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    
    if peak_freqs is not None:
        for peak_freq in peak_freqs:
            peak_index = np.where(np.isclose(freqs, peak_freq, atol=sample_rate / len(fft_samples)))
            if len(peak_index[0]) > 0:
                plt.plot(freqs[peak_index], magnitude[peak_index], 'ro', markersize=10, markerfacecolor='none', markeredgewidth=2)
                plt.annotate('Peak', xy=(freqs[peak_index][0], magnitude[peak_index][0]),
                             xytext=(freqs[peak_index][0], magnitude[peak_index][0] + 10),
                             arrowprops=dict(facecolor='black', shrink=0.05))

    plt.show()

def plot_fft_and_relevant_plots(channel_samples, channel_idx, access_code, lap, sample_rate, channel_bw, min_power_level, max_power_level):
    from sdrfly.demodulators.demodulator_numba import GFSKDemodNumba

    # Initialize the GFSK demodulator
    gfsk_demod = GFSKDemodNumba(kf=0.5)  # Adjust kf value as needed

    # Convert FFT data back to IQ data
    iq_samples = ifft(channel_samples)
    demodulated_signal = gfsk_demod.demodulate(iq_samples)

    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'Channel {channel_idx+1} Visualizations (LAP: {lap}, Access Code: {access_code})', y=1.05)

    # Time Domain Plot
    axs[0].plot(np.real(demodulated_signal), label='Real')
    axs[0].plot(np.imag(demodulated_signal), label='Imag')
    axs[0].set_title(f'Time Domain')
    axs[0].set_xlabel('Sample')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].grid(True)

    # Frequency Domain Plot (FFT)
    fft_demodulated = fft(demodulated_signal)
    freqs = np.fft.fftfreq(len(fft_demodulated), 1 / channel_bw)
    axs[1].plot(np.fft.fftshift(freqs), np.fft.fftshift(20 * np.log10(np.abs(fft_demodulated))))
    axs[1].set_title(f'FFT of Demodulated Signal')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude (dB)')
    axs[1].grid(True)
    axs[1].set_ylim(min_power_level, max_power_level)

    # Spectrogram
    f, t, Sxx = spectrogram(demodulated_signal, fs=channel_bw, nperseg=1024)
    cax = axs[2].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    axs[2].set_title(f'Spectrogram')
    axs[2].set_ylabel('Frequency (Hz)')
    axs[2].set_xlabel('Time (s)')
    fig.colorbar(cax, ax=axs[2], label='Power (dB)')

    # Constellation Diagram
    axs[3].scatter(np.real(demodulated_signal), np.imag(demodulated_signal), s=1)
    axs[3].set_title(f'Constellation Diagram')
    axs[3].set_xlabel('In-phase')
    axs[3].set_ylabel('Quadrature')
    axs[3].grid(True)

    # Eye Diagram
    samples_per_symbol = int(sample_rate / 1e6)  # Assuming 1 MHz symbol rate for simplicity
    eye_data = demodulated_signal[:len(demodulated_signal) // samples_per_symbol * samples_per_symbol]
    eye_data = eye_data.reshape((-1, samples_per_symbol))
    for segment in eye_data:
        axs[4].plot(segment, color='blue', alpha=0.1)
    axs[4].set_title(f'Eye Diagram')
    axs[4].set_xlabel('Sample')
    axs[4].set_ylabel('Amplitude')
    axs[4].grid(True)

    plt.tight_layout()
    plt.show()
