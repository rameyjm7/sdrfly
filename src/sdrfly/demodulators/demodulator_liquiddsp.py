import numpy as np
import ctypes
from sdrfly.demodulators.demodulator_base import DemodulatorBase

# Load the LiquidDSP library
libliquid = ctypes.CDLL('/usr/local/lib/libliquid.so')

# Define LiquidDSP function prototypes
libliquid.freqdem_create.restype = ctypes.c_void_p
libliquid.freqdem_create.argtypes = [ctypes.c_float]

libliquid.freqdem_destroy.restype = None
libliquid.freqdem_destroy.argtypes = [ctypes.c_void_p]

libliquid.freqdem_demodulate.restype = ctypes.c_float
libliquid.freqdem_demodulate.argtypes = [ctypes.c_void_p, ctypes.c_float]

class GFSKDemodLiquidDSP(DemodulatorBase):
    def __init__(self, kf=0.5):
        self.demod = libliquid.freqdem_create(kf)

    def __del__(self):
        libliquid.freqdem_destroy(self.demod)

    def demodulate(self, samples):
        num_samples = len(samples)
        demodulated = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            demodulated[i] = libliquid.freqdem_demodulate(self.demod, samples[i].real)
        return demodulated
