from abc import ABC, abstractmethod


class Layer(ABC):
    _input = None
    _output = None

    n_inputs = None
    n_outputs = None

    input_size = None
    output_size = None

    @abstractmethod
    def initialize_weights(self):
        return NotImplemented

    @abstractmethod
    def forward(self, x):
        return NotImplemented

    @abstractmethod
    def backward(self, error, learning_rate):
        return NotImplemented

    @abstractmethod
    def set_input(self, x):
        return NotImplemented

    def set_n_outputs(self, x):
        return NotImplemented

    @abstractmethod
    def __str__(self):
        return NotImplemented

    @abstractmethod
    def __repr__(self):
        return NotImplemented
