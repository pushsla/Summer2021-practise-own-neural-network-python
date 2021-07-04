import numpy as np

from library.Activation import Activation
from library.Exceptions import DummyClassException, ShapeMismatchException
from library.Neuron import Neuron


class Layer:
    def __init__(self) -> None:
        raise DummyClassException("Layer.Layer")

    def calculate(self, x: np.array) -> np.array:
        pass


class D1FullLayer(Layer):
    def __init__(self, neurons: int, inputs_per_neuron: int, neuron_type: type[Neuron], activation_type: type[Activation], **kwargs) -> None:
        self.inputs_per_neuron = inputs_per_neuron
        self.neurons = neurons
        self.activation_type = activation_type
        self.neuron_type = neuron_type

        self.shape = (self.inputs_per_neuron, self.neurons)

        self.neurons = [
            neuron_type(activation_type, inputs_per_neuron, **kwargs)
            for _ in range(neurons)
        ]

    def calculate(self, x: np.array) -> np.array:
        if len(x.shape) == 1 and x.shape[0] == self.inputs_per_neuron:
            return np.array([neuron.calculate(x) for neuron in self.neurons])
        else:
            raise ShapeMismatchException("D1FullLayer input must be ({},) shape, got {}".format(self.inputs_per_neuron, x.shape))

