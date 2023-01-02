import numpy as np

from src.loss_functions.loss_function_base import LossFunction


# Changed my implementation according to
# https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/losses/losses.py#L110-L222
# Due to numerical instability that occured with my previous implementation
# github coopilot completion
class CrossEntropy(LossFunction):
    def forward(self, y, t):
        self.old_y = y
        self.old_t = t
        eps = np.finfo(np.float64).eps

        return -np.sum(self.old_t * np.log(self.old_y + eps))

    def backward(self):
        return self.old_y - self.old_t
