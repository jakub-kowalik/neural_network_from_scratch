from typing import Tuple

from src.layers.layer_base import Layer
import numpy as np


# https://blog.ca.meron.dev/Vectorized-CNN/
# Author: Cameron White
# Date: 27.12.2022
# Comment: Code from blog post is used with minor changed to match implementation.
#          Mainly used to compare performance and correctness of my implementation.
class Conv2DVectorized(Layer):
    def __init__(self, input_size=None, n_outputs=1, filter_size=3, stride=1, padding=0):
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

    def initialize_weights(self):
        if not (self.n_inputs is None or self.n_outputs is None):
            self.weights = 1e-3 * np.random.randn(self.n_outputs, self.n_inputs, self.filter_size,
                                                  self.filter_size)
            self.bias = np.zeros(self.n_outputs)

            self.output_size = self.calculate_output_size()

    def calculate_output_size(self):
        c, h, w = self.input_size
        out_h = (h - self.filter_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.filter_size + 2 * self.padding) // self.stride + 1

        return self.n_outputs, out_h, out_w

    def set_input(self, x: Tuple[int, int, int]):
        self.input_size = x
        self.n_inputs = x[0]

        self.initialize_weights()

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
        out += self.bias[None, :, None, None]

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

    def __str__(self):
        return self.__class__.__name__ \
            + f"(input size: {self.input_size}, " \
            + f"output channels: {self.n_outputs}, " \
            + f"filter size: {self.filter_size}, " \
            + f"stride: {self.stride}, " \
            + f"padding: {self.padding})"

    def __repr__(self):
        return self.__str__()
