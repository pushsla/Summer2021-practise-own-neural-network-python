import numpy as np

from library.Activation import Activation
from library.Exceptions import DummyClassException


class Neuron:
    def __init__(self, activation_type: type[Activation], inputs: int, thresh: float = 0, **kwargs) -> None:
        raise DummyClassException("Neuron.Neuron")

    def calculate(self, x: np.array) -> float:
        pass


class WeightedSumNeuron(Neuron):
    def __init__(self, activation_type: type[Activation], inputs: int, thresh: float = 0, **kwargs) -> None:
        self.activation = activation_type
        self.activation_instance = activation_type(**kwargs)
        self.inputs = inputs
        self.thresh = thresh
        self.weights = np.random.rand(self.inputs)

    def calculate(self, x: np.array) -> float:
        weighted_x = x * self.weights
        weighted_sum = np.sum(weighted_x) + self.thresh
        return self.activation_instance.value(weighted_sum)