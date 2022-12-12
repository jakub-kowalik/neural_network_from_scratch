from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @staticmethod
    @abstractmethod
    def forward(y, t):
        return NotImplemented

    @staticmethod
    @abstractmethod
    def backward():
        return NotImplemented


# github coopilot completion
class MeanSquareError(LossFunction):
    def forward(self, y, t):
        self.input_y = y
        self.input_t = t
        return np.square(np.subtract(y, t)).mean()

    def backward(self):
        return 2 * (self.input_y - self.input_t) / self.input_y.size


class TestError(LossFunction):
    def forward(self, y, t):
        self.input_y = y
        self.input_t = t
        return np.mean(np.abs(y - t))

    def backward(self):
        return 2 * 1 * (self.input_y - self.input_t)


# github coopilot completion
#https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/losses/losses.py#L110-L222
class CrossEntropy(LossFunction):
    def forward(self, y, t):
        self.old_y = y
        self.old_t = t
        eps = np.finfo(float).eps

        return -np.sum(self.old_t * np.log(self.old_y + eps))

    def backward(self):
        return self.old_y - self.old_t
        # return -self.old_t / self.old_y  # + (1 - self.old_t) / (1 - self.old_y)


# old
def mean_square_error(y, t):
    return np.mean((y - t) ** 2)


def mean_square_error_derivative(y, t):
    return 2 * (y - t)


# https://math.stackexchange.com/questions/2843505/derivative-of-softmax-without-cross-entropy
def cross_entropy(y, t):
    return np.sum(-t * np.log(y))


def cross_entropy_derivatives(y, t):
    return -t / y