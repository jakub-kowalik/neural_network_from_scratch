import numpy as np

from src.activation_functions.activation_base import ActivationFunction


class Softmax(ActivationFunction):
    # https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/layers/layers.py#L2192-L2349
    # probably more numerically stable than my previous implementation
    def forward(self, x):
        e_X = np.exp(x - np.max(x, axis=1, keepdims=True))
        self._output = e_X / e_X.sum(axis=1, keepdims=True)
        return self._output

    # https://math.stackexchange.com/questions/2843505/derivative-of-softmax-without-cross-entropy
    # currently not being used anyway
    def backward(self, x):
        return 1.0  # 1 to retain error on backpropagation multiply

        # s = self._output.reshape(-1, 1)
        # self._error = np.diagflat(s) - np.dot(s, s.T)
        # return self._error
        # return np.exp(self._output) / np.sum(np.exp(self._output)) \
        #     * np.exp(1 - self._output) / np.sum(np.exp(1 - self._output))

        # self._error = self._output - x
        # return self._error