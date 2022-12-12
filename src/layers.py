from abc import abstractmethod, ABC

import numpy as np

from src.activation_functions import ActivationFunction


class Layer(ABC):
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
        self.x = x
        return self.activation_function.forward(x)

    def backward(self, error, learning_rate):
        return error * self.activation_function.backward(self.x)


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.input = None
        self.output = None
        self.weights = np.random.normal(0, 0.01, (n_inputs, n_outputs)).astype(np.float32)
        # @TODO check how to initialize better

        self.bias = np.zeros((1, n_outputs))

    def forward(self, x):
        self.input = x
        self.output = (self.input @ self.weights) + self.bias

        return self.output

    def backward(self, error, learning_rate):
        input_error = error @ self.weights.T

        self.weights -= learning_rate * (self.input.T @ error)
        self.bias -= learning_rate * np.mean(error, axis=0)

        return input_error
