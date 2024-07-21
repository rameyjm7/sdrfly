import numpy as np

def detect_peaks(samples, threshold=0.5):
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(np.abs(samples), height=threshold)
    return peaks

def extract_lap_and_access_code(demodulated_signal, peaks):
    results = []
    for peak in peaks:
        # Assuming access code is in the first 72 bits (9 bytes) following the peak
        access_code_bits = demodulated_signal[peak:peak + 72]
        access_code = ''.join(['1' if bit > 0 else '0' for bit in access_code_bits])
        access_code_hex = hex(int(access_code, 2))

        # Assuming LAP is the last 24 bits (3 bytes) of the access code
        lap_bits = access_code_bits[-24:]
        lap = ''.join(['1' if bit > 0 else '0' for bit in lap_bits])
        lap_hex = format(int(lap, 2), '06X')
        lap_formatted = ':'.join([lap_hex[i:i+2] for i in range(0, len(lap_hex), 2)])

        # Ignore LAPs that are 00:00:00
        if lap_formatted != '00:00:00':
            results.append((access_code_hex, lap_formatted))
    
    return results

def probe_devices():
    import SoapySDR
    results = SoapySDR.Device.enumerate()
    for result in results:
        print(f"Found device: {result}")
