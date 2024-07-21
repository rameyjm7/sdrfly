import numpy as np
from numba import jit
from sdrfly.demodulators.demodulator_base import DemodulatorBase

class GFSKDemodNumba(DemodulatorBase):
    def __init__(self, kf=0.5):
        self.kf = kf

    @staticmethod
    @jit(nopython=True)
    def gfsk_demodulate(samples, kf):
        num_samples = len(samples)
        demodulated = np.zeros(num_samples, dtype=np.float32)
        previous_sample = 0.0

        for i in range(num_samples):
            demodulated[i] = np.angle(samples[i] * np.conj(previous_sample)) / kf
            previous_sample = samples[i]

        return demodulated

    def demodulate(self, samples):
        return self.gfsk_demodulate(samples, self.kf)
