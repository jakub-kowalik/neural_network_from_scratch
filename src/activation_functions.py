import math

import numpy as np
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


class Sigmoid(ActivationFunction):
    # if my implementation wont work
    # https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth

    def forward(self, x):
        self._output = 1 / (1 + np.exp(-x))
        return self._output

    def backward(self, x, *args):
        self._error = 1 / (1 + np.exp(-self._output)) * 1 / (1 + np.exp(-(1 - self._output)))
        return self._error


# https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
class ReLU(ActivationFunction):
    def forward(self, x):
        self._output = x * (x > 0)
        return self._output

    def backward(self, x):
        self._error = 1. * (self._output > 0)
        return self._error


class Softmax(ActivationFunction):
    # https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/layers/layers.py#L2192-L2349
    def forward(self, x):
        # self._output = np.exp(x) / np.sum(np.exp(x))
        # return self._output
        e_X = np.exp(x - np.max(x, keepdims=True))
        return e_X / e_X.sum(keepdims=True)

    # https://math.stackexchange.com/questions/2843505/derivative-of-softmax-without-cross-entropy
    def backward(self, x):
        return 1
        s = self._output.reshape(-1, 1)
        self._error = np.diagflat(s) - np.dot(s, s.T)
        return self._error
        # return np.exp(self._output) / np.sum(np.exp(self._output)) \
        #     * np.exp(1 - self._output) / np.sum(np.exp(1 - self._output))

        # self._error = self._output - x
        # return self._error

