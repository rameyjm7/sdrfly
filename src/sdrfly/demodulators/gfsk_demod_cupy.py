import cupy as cp
import numpy as np
from bluetooth_demod.demodulators.demodulator_base import DemodulatorBase

class GFSKDemodCuPy(DemodulatorBase):
    def __init__(self, kf):
        self.kf = kf

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        samples_gpu = cp.asarray(samples)
        demodulated = cp.zeros(samples_gpu.size, dtype=cp.float32)
        
        # Initialize phase accumulator
        phase = cp.float32(0.0)
        
        # GFSK Demodulation
        for i in range(1, samples_gpu.size):
            # Compute phase difference
            delta_phase = cp.angle(samples_gpu[i] * cp.conj(samples_gpu[i - 1]))
            phase += delta_phase
            
            # Normalize phase
            phase = cp.mod(phase, 2 * cp.pi)
            
            # Convert phase to frequency deviation
            demodulated[i] = phase / self.kf
        
        return cp.asnumpy(demodulated)
