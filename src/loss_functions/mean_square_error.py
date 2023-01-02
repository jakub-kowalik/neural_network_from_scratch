import numpy as np

from src.loss_functions.loss_function_base import LossFunction


#github copilot completion
class MeanSquareError(LossFunction):
    def forward(self, y, t):
        self.input_y = y
        self.input_t = t
        return np.square(np.subtract(y, t)).mean()

    def backward(self):
        return 2 * (self.input_y - self.input_t) / len(self.input_y)
