import numpy as np

from src.loss_functions.loss_function_base import LossFunction


class CrossEntropy(LossFunction):
    # github coopilot completion
    #https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/losses/losses.py#L110-L222
    def forward(self, y, t):
        self.old_y = y
        self.old_t = t
        eps = np.finfo(float).eps

        return -np.sum(self.old_t * np.log(self.old_y + eps))

    def backward(self):
        return self.old_y - self.old_t
        # return -self.old_t / self.old_y  # + (1 - self.old_t) / (1 - self.old_y)
