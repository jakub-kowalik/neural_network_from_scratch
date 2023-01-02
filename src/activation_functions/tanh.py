import numpy as np

from src.activation_functions.activation_base import ActivationFunction


# https://pl.wikipedia.org/wiki/Funkcje_hiperboliczne
# apparently not numerically stable
# github copilot completion
class Tanh(ActivationFunction):
    def forward(self, x):
        self._output = np.tanh(x)
        return self._output

    def backward(self, x):
        self._error = 1 - np.tanh(self._output) ** 2
        return self._error
