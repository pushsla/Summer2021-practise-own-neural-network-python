import numpy as np

from library.DeprecatedActivation import Activation, SigmoidActivation
from library.DeprecatedExceptions import DummyClassException


class Neuron:
    def __init__(self, inputs: int, activation_type: type[Activation] = SigmoidActivation, thresh: float = 0) -> None:
        raise DummyClassException("Neuron.Neuron")
        self.activation_type = activation_type
        self.activation_instance = activation_type()
        self.inputs = inputs
        self.thresh = thresh
        self.weights = np.random.rand(self.inputs)

    def calculate(self, x: np.array) -> float:
        pass


class WeightedSumNeuron(Neuron):
    def __init__(self, inputs: int, activation_type: type[Activation] = SigmoidActivation, thresh: float = 0) -> None:
        self.activation_type = activation_type
        self.activation_instance = activation_type()
        self.inputs = inputs
        self.thresh = thresh
        self.weights = np.random.rand(self.inputs)

    def calculate(self, x: np.array) -> float:
        weighted_x = x * self.weights
        weighted_sum = np.sum(weighted_x) + self.thresh
        return self.activation_instance.value(weighted_sum)