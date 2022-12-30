import numpy as np

from src.loss_functions.loss_function_base import LossFunction


class MeanAbsoluteError(LossFunction):
    def forward(self, y, t):
        self.input_y = y
        self.input_t = t
        return np.mean(np.abs(y - t))

    def backward(self):
        return 2 * 1 * (self.input_y - self.input_t)
