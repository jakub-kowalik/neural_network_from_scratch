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


def getWindows(input, output_size, kernel_size, padding=0, stride=1, dilate=0):
    working_input = input
    working_pad = padding
    # dilate the input if necessary
    if dilate != 0:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

    # pad the input if necessary
    if working_pad != 0:
        working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))

    in_b, in_c, out_h, out_w = output_size
    out_b, out_c, _, _ = input.shape
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

    return np.lib.stride_tricks.as_strided(
        working_input,
        (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
        (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
    )

class Conv2D(Layer):
    def __init__(self, n_inputs, n_outputs, filter_size, stride=1, padding=0):
        super().__init__()
        self._input = None
        self._output = None
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        stddev = np.sqrt(2 / (n_inputs + n_outputs + filter_size + filter_size))
        self.weights = np.random.normal(0, stddev, (n_outputs, n_inputs, filter_size, filter_size)).astype(np.float32)
        # self.weights = np.random.normal(0, 0.01, (n_filters, filter_size, filter_size)).astype(np.float32)
        self.bias = np.zeros((1, n_outputs)).astype(np.float32)

    def forward(self, x):
        self._input = x

        input_batch_size, input_channels, input_height, input_width = x.shape

        # print(x.shape)
       # padded\
        padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        batch_size, n_channels, height, width = padded.shape  # padded.shape

        self._output = np.empty((  # empty should be faster than zeros
            input_batch_size,
            self.n_outputs,
            ((input_height - self.filter_size + (2 * self.padding)) // self.stride) + 1,
            ((input_width - self.filter_size + (2 * self.padding)) // self.stride) + 1
        ))

        out_samples, out_channels, out_height, out_width = self._output.shape

        # for b in range(out_samples):
        for f in range(self.n_outputs):
            for h in range(out_height):
                for w in range(out_width):
                    self._output[:, f, h, w] = np.sum(
                        padded[:, :,
                        h * self.stride:h * self.stride + self.filter_size,
                        w * self.stride:w * self.stride + self.filter_size
                        ] * self.weights[f]) + self.bias[0][f]

        return self._output

    def backward(self, error, learning_rate):
        batch_size, n_channels, height, width = self._input.shape
        out_samples, out_channels, out_height, out_width = error.shape

        error_padded = np.pad(error, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        input_error = np.empty(self._input.shape)
        out_samples, out_channels, out_height, out_width = self._input.shape

        padded = np.pad(self._input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        # # for b in range(out_samples):
        # for f in range(out_channels):
        #     for h in range(out_height):
        #         for w in range(out_width):
        #             input_error[:, f, h, w] \
        #                 = np.sum(error_padded[:, :, h:h + self.filter_size, w:w + self.filter_size]
        #                          * self.weights)

        # # print(input_error.shape)
        # print(error_padded.shape, 'x')
        # print(self._input.shape)
        # print(input_error.shape)
        # print(error.shape, 'x')
        # print(padded.shape)

        # old_weights = self.weights.copy()
        for c in range(self.n_inputs):
            for f in range(self.n_outputs):
                for h in range(self.filter_size):
                    for w in range(self.filter_size):
                        self.weights[f, :, h, w] -= learning_rate * np.sum(
                            padded[:, c,
                            h * self.stride:h * self.stride + error.shape[-2],
                            w * self.stride:w * self.stride + error.shape[-1]] * error)

        self.bias -= learning_rate * np.sum(error.astype(np.float32), axis=(0, 2, 3))

        # print(self.weights - old_weights)
        # print("-------------")

        return input_error