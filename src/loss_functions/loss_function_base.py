from abc import ABC, abstractmethod


class LossFunction(ABC):

    input_y = None
    input_t = None

    @abstractmethod
    def forward(self, y, t):
        return NotImplemented

    @abstractmethod
    def backward(self):
        return NotImplemented
