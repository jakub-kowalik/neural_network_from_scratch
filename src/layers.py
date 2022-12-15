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


class Conv2D(Layer):
    def __init__(self, n_filters, filter_size, stride=1, padding=0):
        super().__init__()
        self._input = None
        self._output = None
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        stddev = np.sqrt(2 / (n_filters + filter_size + filter_size))
        self.weights = np.random.normal(0, stddev, (n_filters, filter_size, filter_size)).astype(np.float32)
        # self.weights = np.random.normal(0, 0.01, (n_filters, filter_size, filter_size)).astype(np.float32)
        self.bias = np.zeros((1, n_filters)).astype(np.float32)

    def forward(self, x):
        self._input = x

        # print(x.shape)
       # padded\
        padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        batch_size, n_channels, height, width = padded.shape  # padded.shape

        self._output = np.zeros((
            batch_size,
            self.n_filters,
            ((x.shape[2] - self.filter_size + 2 * self.padding) // self.stride) + 1,
            ((x.shape[2] - self.filter_size + 2 * self.padding) // self.stride) + 1
        ))

        out_samples, out_channels, out_height, out_width = self._output.shape

        for b in range(out_samples):
            for f in range(self.n_filters):
                for h in range(out_height):
                    for w in range(out_width):
                        self._output[b, f, h, w] = np.sum(
                            padded[
                            b,
                            :,
                            h * self.stride:h * self.stride + self.filter_size,
                            w * self.stride:w * self.stride + self.filter_size]
                            *
                            self.weights[f] + self.bias[0][f]  # no kernel flip np.flip()
                        )

        self._input = padded

        return self._output

    def backward(self, error, learning_rate):
        batch_size, n_channels, height, width = self._input.shape
        out_samples, out_channels, out_height, out_width = error.shape

        error_padded = np.pad(error, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        input_error = np.zeros(error_padded.shape)
        out_samples, out_channels, out_height, out_width = error.shape

        for b in range(out_samples):
            for f in range(self.n_filters):
                for h in range(out_height):
                    for w in range(out_width):
                        input_error[
                        b,
                        f,
                        h,
                        w] = np.sum(error_padded[b, f, h:h + self.filter_size, w:w + self.filter_size]
                                    * self.weights[f])

        for f in range(self.n_filters):
            for h in range(self.filter_size):
                for w in range(self.filter_size):
                    self.weights[:, h, w] -= learning_rate * np.sum(
                        self._input[:, :, h:h + error.shape[-2], w:w + error.shape[-1]] * error)

        self.bias -= learning_rate * np.sum(error)

        return input_error
