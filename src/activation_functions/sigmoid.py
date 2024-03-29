import numpy as np

from src.activation_functions.activation_base import ActivationFunction


# if my implementation wont work due to numerical instability, I will use this one
# https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
class Sigmoid(ActivationFunction):
    def forward(self, x):
        self._output = 1 / (1 + np.exp(-x))
        return self._output

    def backward(self, x, *args):
        self._error = (1 / (1 + np.exp(-self._output))) * (
            1 - (1 / (1 + np.exp(-self._output)))
        )
        return self._error
