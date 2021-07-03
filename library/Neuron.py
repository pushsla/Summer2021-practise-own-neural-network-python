import numpy as np

from library.Activation import Activation
from library.Exceptions import DummyClassException


class Neuron:
    def __init__(self, activation: Activation, inputs: int = 2, thresh: float = 0) -> None:
        raise DummyClassException("Neuron.Neuron")

    def calculate(self, x: np.array):
        pass


class WeightedSumNeuron(Neuron):
    def __init__(self, activation: Activation, inputs: int = 2, thresh: float = 0) -> None:
        self.activation = activation
        self.inputs = inputs
        self.thresh = thresh
        self.weights = np.random.rand(self.inputs)

    def calculate(self, x: np.array):
        weighted_x = x * self.weights
        weighted_sum = np.sum(weighted_x) + self.thresh
        return self.activation.value(weighted_sum)