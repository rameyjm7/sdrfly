import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class ChannelizerBase:
    def __init__(self, num_channels=10, channel_bw=1e6, sample_rate=10e6):
        self.num_channels = num_channels
        self.channel_bw = channel_bw
        self.sample_rate = sample_rate

    def channelize(self, samples):
        raise NotImplementedError("This method should be implemented by subclasses")
