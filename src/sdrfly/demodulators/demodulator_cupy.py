import cupy as cp
from bluetooth_demod.demodulators.demodulator_base import DemodulatorBase

class GFSKDemod(DemodulatorBase):
    def __init__(self, kf=0.5):
        self.kf = kf

    def demodulate(self, samples):
        num_samples = len(samples)
        demodulated = cp.zeros(num_samples, dtype=cp.float32)
        previous_sample = cp.array(0.0, dtype=cp.complex64)
        
        samples_gpu = cp.asarray(samples, dtype=cp.complex64)
        
        for i in range(num_samples):
            demodulated[i] = cp.angle(samples_gpu[i] * cp.conj(previous_sample)) / self.kf
            previous_sample = samples_gpu[i]
        
        return demodulated
