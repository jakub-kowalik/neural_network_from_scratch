from typing import Tuple

import numpy as np

from src.layers.layer_base import Layer


# resources used:
# https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html
# https://stackoverflow.com/questions/61264959/how-to-zero-pad-rgb-image
# https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html
# https://johnwlambert.github.io/conv-backprop/
class Conv2D(Layer):
    def __init__(
        self, input_size=None, n_outputs=1, filter_size=3, stride=1, padding=0
    ):
        """

        :param input_size: convention input size (c, h, w) e.g. (3, 32, 32)
        :param n_outputs:
        :param filter_size:
        :param stride:
        :param padding:
        """
        super().__init__()
        self._input = None
        self._output = None

        self.input_size = input_size

        if input_size is None:
            self.n_inputs = None
        else:
            self.n_inputs = input_size[0]

        self.n_outputs = n_outputs
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.initialize_weights()

    def set_input(self, x: Tuple[int, int, int]):
        self.input_size = x
        self.n_inputs = x[0]

        self.initialize_weights()

    def set_n_outputs(self, x):
        self.n_outputs = x

    # @TODO create package for weight initialization
    def initialize_weights(self):
        if not (self.n_inputs is None or self.n_outputs is None):
            stddev = np.sqrt(2 / (self.n_inputs + self.n_outputs))
            self.weights = np.random.normal(
                0,
                stddev,
                (self.n_inputs, self.n_outputs, self.filter_size, self.filter_size),
            )
            # self.weights = np.random.normal(0, 0.01, (n_filters, filter_size, filter_size)).astype(np.float32)
            self.bias = np.zeros((1, self.n_outputs))

            self.output_size = self.calculate_output_size()

    def calculate_output_size(self):
        return (  # empty should be faster than zeros
            self.n_outputs,
            (
                (self.input_size[1] - self.filter_size + (2 * self.padding))
                // self.stride
            )
            + 1,
            (
                (self.input_size[2] - self.filter_size + (2 * self.padding))
                // self.stride
            )
            + 1,
        )

    def forward(self, x):

        self._input = x

        input_batch_size, input_channels, input_height, input_width = x.shape

        # B C H W -> 0 on B, 0 on C, padding on H, padding on W
        padded = np.pad(
            x,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            constant_values=0,
        )

        batch_size, n_channels, height, width = padded.shape  # padded.shape

        self._output = np.empty(
            (  # empty should be faster than zeros
                input_batch_size,
                self.n_outputs,
                ((input_height - self.filter_size + (2 * self.padding)) // self.stride)
                + 1,
                ((input_width - self.filter_size + (2 * self.padding)) // self.stride)
                + 1,
            )
        )

        out_samples, out_channels, out_height, out_width = self._output.shape

        # for b in range(out_samples):
        for o in range(self.n_outputs):
            for h in range(out_height):
                for w in range(out_width):
                    self._output[:, o, h, w] = (
                        np.sum(
                            padded[
                                :,
                                :,
                                h * self.stride : h * self.stride + self.filter_size,
                                w * self.stride : w * self.stride + self.filter_size,
                            ]
                            * self.weights[:, o],
                            axis=(1, 2, 3),
                        )
                        + self.bias[0][o]
                    )

        return self._output

    def backward(self, error, learning_rate):
        batch_size, n_channels, height, width = self._input.shape
        out_samples, out_channels, out_height, out_width = error.shape

        padded = np.pad(
            self._input,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            constant_values=0,
        )

        pdd = (self._input.shape[2] - error.shape[2]) // 2 + self.filter_size // 2
        error_padded = np.pad(
            error, ((0, 0), (0, 0), (pdd, pdd), (pdd, pdd)), constant_values=0
        )

        input_error = np.zeros_like(self._input)

        out_samples, out_channels, out_height, out_width = input_error.shape

        for i in range(self.n_inputs):
            for o in range(self.n_outputs):
                for h in range(out_height):
                    for w in range(out_width):
                        input_error[:, i, h, w] += np.sum(
                            self.weights[i, o]
                            * error_padded[
                                :, o, h : h + self.filter_size, w : w + self.filter_size
                            ],
                            axis=(1, 2),
                        )

        # old implementation using scipy correlate2d
        # very slow

        # for b in range(out_samples):
        #     for i in range(self.n_inputs):
        #         for o in range(self.n_outputs):
        #             self.weights[i, o] -= learning_rate \
        #                                   * scipy.signal.correlate2d(self._input[b][i], error[b][o], mode='valid')

        error_batch_size, error_channels, error_height, error_width = error.shape

        for i in range(self.n_inputs):
            for o in range(self.n_outputs):
                for h in range(self.filter_size):
                    for w in range(self.filter_size):
                        self.weights[i, o, h, w] -= learning_rate * np.sum(
                            padded[
                                :,
                                i,
                                h * self.stride : h * self.stride + error_height,
                                w * self.stride : w * self.stride + error_width,
                            ]
                            * error[:, o, :, :]
                        )

        self.bias[0, :] -= learning_rate * np.sum(error, axis=(0, 2, 3))

        return input_error

    def __str__(self):
        return (
            self.__class__.__name__
            + f"(input size: {self.input_size}, "
            + f"output size: {self.output_size}, "
            + f"filter size: {self.filter_size}, "
            + f"stride: {self.stride}, "
            + f"padding: {self.padding}"
            + f")"
        )

    def __repr__(self):
        return self.__str__()
