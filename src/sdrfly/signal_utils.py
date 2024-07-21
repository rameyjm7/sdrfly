import numpy as np

def generate_cw_tone(freq, sample_rate, num_samples):
    t = np.arange(num_samples) / sample_rate
    signal = np.exp(2j * np.pi * freq * t)
    return signal

def generate_two_cw_tones(freq1, freq2, sample_rate, num_samples):
    t = np.arange(num_samples) / sample_rate
    signal1 = np.exp(2j * np.pi * freq1 * t)
    signal2 = np.exp(2j * np.pi * freq2 * t)
    return signal1 + signal2
