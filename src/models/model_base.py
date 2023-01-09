from abc import ABC, abstractmethod
from typing import Tuple, Union, List
import matplotlib.pyplot as plt


class Model(ABC):
    @abstractmethod
    def __init__(self, input_size: Union[int, Tuple[int, int, int]], output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []

        self.training_losses = []
        self.validation_losses = []

        self.training_accuracies = []
        self.validation_accuracies = []

    @abstractmethod
    def predict(self, x, training=False):
        return NotImplemented

    @abstractmethod
    def fit(self, x, y):
        return NotImplemented

    def summary(self):
        print(str(self.__class__.__name__) + " {")
        for layer in self.layers:
            print("\t" + str(layer))
        print("}")

    def plot_history(self, parameters: List[str] = ["loss"]):
        if parameters is None:
            raise ValueError("parameters must not be None")

        plt.title("Model History")
        if "loss" in parameters:
            plt.plot(self.training_losses, label="train loss")
            plt.plot(self.validation_losses, label="val loss")
        if "accuracy" in parameters:
            plt.plot(self.training_accuracies, label="train accuracy", linestyle="--")
            plt.plot(self.validation_accuracies, label="val accuracy", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
