from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    _output = None
    _error = None

    @abstractmethod
    def forward(self, x):
        raise NotImplemented

    @abstractmethod
    def backward(self, x):
        raise NotImplemented

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__
