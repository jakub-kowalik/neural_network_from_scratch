from abc import abstractmethod, ABC

import numpy as np
import scipy

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
    def __init__(self, n_inputs, n_outputs, filter_size, stride=1, padding=0):
        super().__init__()
        self._input = None
        self._output = None
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        stddev = np.sqrt(2 / (n_outputs + filter_size + filter_size))
        self.weights = np.random.normal(0, stddev, (n_inputs, n_outputs, filter_size, filter_size)).astype(np.float32)
        # self.weights = np.random.normal(0, 0.01, (n_filters, filter_size, filter_size)).astype(np.float32)
        self.bias = np.zeros((1, n_outputs)).astype(np.float32)

    def forward(self, x):
        # resources used:
        # https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html

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
        for o in range(self.n_outputs):
            for h in range(out_height):
                for w in range(out_width):
                    self._output[:, o, h, w] = np.sum(
                        padded[:, :,
                        h * self.stride:h * self.stride + self.filter_size,
                        w * self.stride:w * self.stride + self.filter_size
                        ] * self.weights[:, o], axis=(1, 2, 3)) + self.bias[0][o]

        return self._output

    def backward(self, error, learning_rate):
        # resources used:
        # https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html
        # https://johnwlambert.github.io/conv-backprop/

        batch_size, n_channels, height, width = self._input.shape
        out_samples, out_channels, out_height, out_width = error.shape

        error_padded = error

        pdd = (self._input.shape[2] - error_padded.shape[2]) // 2
        error_padded = np.pad(error,
                              ((0, 0), (0, 0), (pdd, pdd), (pdd, pdd)),
                              constant_values=1)

        input_error = np.zeros_like(self._input).astype(np.float32)

        out_samples, out_channels, out_height, out_width = self._output.shape

        for i in range(self.n_inputs):
            for o in range(self.n_outputs):
                for h in range(out_height):
                    for w in range(out_width):
                        input_error[:, i,
                        h * self.stride:h * self.stride + self.filter_size,
                        w * self.stride:w * self.stride + self.filter_size] \
                            += self.weights[i, o] * error_padded[:, o,
                                                 h * self.stride:h * self.stride + self.filter_size,
                                                 w * self.stride:w * self.stride + self.filter_size]

        for b in range(out_samples):
            for i in range(self.n_inputs):
                for o in range(self.n_outputs):
                    self.weights[i, o] -= learning_rate \
                                          * scipy.signal.correlate2d(self._input[b][i], error[b][o], mode='valid')

        self.bias[0, :] -= learning_rate * np.sum(error, axis=(0, 2, 3))

        return input_error


# https://blog.ca.meron.dev/Vectorized-CNN/
class Conv2DVectorized(Layer):
    def __init__(self, n_inputs, n_outputs, filter_size, stride=1, padding=0):
        super().__init__()
        self._input = None
        self._output = None
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        stddev = np.sqrt(2 / (n_outputs + filter_size + filter_size))
        self.weights = np.random.normal(0, stddev, (n_outputs, n_inputs, filter_size, filter_size)).astype(np.float32)
        # self.weights = np.random.normal(0, 0.01, (n_filters, filter_size, filter_size)).astype(np.float32)
        self.bias = np.zeros((1, n_outputs)).astype(np.float32)

    def getWindows(self, input, output_size, kernel_size, padding=0, stride=1, dilate=0):
        working_input = input
        working_pad = padding
        # dilate the input if necessary
        if dilate != 0:
            working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
            working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

        # pad the input if necessary
        if working_pad != 0:
            working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)),
                                   mode='constant', constant_values=(0.,))

        in_b, in_c, out_h, out_w = output_size
        out_b, out_c, _, _ = input.shape
        batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

        return np.lib.stride_tricks.as_strided(
            working_input,
            (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
            (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
        )

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters.
        """

        n, c, h, w = x.shape
        out_h = (h - self.filter_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.filter_size + 2 * self.padding) // self.stride + 1

        windows = self.getWindows(x, (n, c, out_h, out_w), self.filter_size, self.padding, self.stride)

        out = np.einsum('bihwkl,oikl->bohw', windows, self.weights)

        # add bias to kernels
        out += self.bias[0][None, :, None, None]

        self.cache = x, windows
        return out

    def backward(self, dout, learning_rate):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: dx, dw, and db relative to this module
        """
        x, windows = self.cache

        padding = self.filter_size - 1 if self.padding == 0 else self.padding

        dout_windows = self.getWindows(dout, x.shape, self.filter_size, padding=padding, stride=1,
                                       dilate=self.stride - 1)
        rot_kern = np.rot90(self.weights, 2, axes=(2, 3))

        db = np.sum(dout, axis=(0, 2, 3))
        dw = np.einsum('bihwkl,bohw->oikl', windows, dout)
        dx = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)

        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

        return dx
