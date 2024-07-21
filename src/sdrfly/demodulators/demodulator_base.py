from abc import ABC, abstractmethod

class DemodulatorBase(ABC):
    @abstractmethod
    def demodulate(self, samples):
        pass
