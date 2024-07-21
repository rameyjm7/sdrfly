
import numpy as np
import SoapySDR
import threading
from sdrfly.sdr.sdr_base import SDR
from sdrfly.sdr.sdr_hackrf import HackRFSdr
from sdrfly.sdr.sdr_sidekiq import SidekiqSdr

class SDRGeneric:
    def __new__(cls, sdr_type, *args, **kwargs):
        if sdr_type == "hackrf":
            from sdrfly.sdr.sdr_hackrf import HackRFSdr
            return HackRFSdr(*args, **kwargs)
        elif sdr_type == "sidekiq":
            from sdrfly.sdr.sdr_sidekiq import SidekiqSdr
            return SidekiqSdr(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported SDR type: {sdr_type}")

# Example usage:
# sdr = SDRGeneric("hackrf", center_freq=915e6, sample_rate=10e6, bandwidth=5e6, gain=20, size=1024)
# sdr = SDRGeneric("sidekiq", center_freq=915e6, sample_rate=10e6, bandwidth=5e6, gain=20, size=1024)
