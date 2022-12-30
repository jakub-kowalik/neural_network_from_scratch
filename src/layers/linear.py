import numpy as np

from src.layers.layer_base import Layer


class Linear(Layer):
    def __init__(
            self,
            n_inputs: int = None,
            n_outputs: int = None
    ):
        super().__init__()
        self._input = None
        self._output = None

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.input_size = self.n_inputs
        self.output_size = self.n_outputs

        self.initialize_weights()
        # https://stackoverflow.com/questions/62249084/what-is-the-numpy-equivalent-of-tensorflow-xavier-initializer-for-cnn
        # self.weights = np.random.uniform(-0.005, 0.005, (n_inputs, n_outputs)).astype(np.float32)

    def initialize_weights(self):
        if not (self.n_inputs is None or self.n_outputs is None):
            # xavier from gpt
            stddev = np.sqrt(2 / (self.n_inputs + self.n_outputs))
            self.weights = np.random.normal(0, stddev, (self.n_inputs, self.n_outputs)).astype(np.float32)

            self.bias = np.zeros((1, self.n_outputs))

    def forward(self, x):
        self._input = x
        self._output = (self._input @ self.weights) + self.bias

        return self._output

    def backward(self, error, learning_rate):
        input_error = error @ self.weights.T

        self.weights -= learning_rate * (self._input.T @ error)
        self.bias -= learning_rate * np.sum(error, axis=0, keepdims=True)

        return input_error

    def set_input(self, x):
        self.input_size = x
        self.n_inputs = x
        self.output_size = self.n_outputs

    def set_n_outputs(self, x):
        pass

    def __str__(self):
        return self.__class__.__name__ + f"(inputs: {self.n_inputs}, outputs: {self.n_outputs})"

    def __repr__(self):
        return self.__str__()
