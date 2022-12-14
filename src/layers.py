from abc import abstractmethod, ABC

import numpy as np

from src.activation_functions import ActivationFunction


class Layer(ABC):

    _input = None
    _output = None

    @abstractmethod
    def forward(self, x):
        return NotImplemented

    @abstractmethod
    def backward(self, error, learning_rate):
        return NotImplemented


class Activation(Layer):
    def __init__(self, activation_function: ActivationFunction):
        self.activation_function = activation_function

    def forward(self, x):
        self._input = x
        return self.activation_function.forward(x)

    def backward(self, error, learning_rate):
        return error * self.activation_function.backward(self._input)


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self._input = None
        self._output = None
        # xavier from gpt
        stddev = np.sqrt(2 / (n_inputs + n_outputs))
        self.weights = np.random.normal(0, stddev, (n_inputs, n_outputs)).astype(np.float32)
        # https://stackoverflow.com/questions/62249084/what-is-the-numpy-equivalent-of-tensorflow-xavier-initializer-for-cnn
        # self.weights = np.random.uniform(-0.005, 0.005, (n_inputs, n_outputs)).astype(np.float32)
        # @TODO check how to initialize better

        self.bias = np.zeros((1, n_outputs))

    def forward(self, x):
        self._input = x
        self._output = (self._input @ self.weights) + self.bias

        return self._output

    def backward(self, error, learning_rate):
        input_error = error @ self.weights.T

        self.weights -= learning_rate * (self._input.T @ error)
        self.bias -= learning_rate * np.sum(error, axis=0, keepdims=True)

        return input_error


class Flatten(Layer):
    # github copilot completion
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, error, learning_rate):
        return error.reshape(self.input_shape)
