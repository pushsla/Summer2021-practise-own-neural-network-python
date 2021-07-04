from library.Exceptions import ShapeMismatchException, LibraryException
from library.Layer import D1FullLayer
import numpy as np


class D1FullNetwork:
    def __init__(self, input_size: int) -> None:
        self.shape = (input_size,)
        self.layers_count = 0
        self.layers = []

    def add_layer(self, neurons: int, **kwargs):
        layer_inputs = self.layers[-1].shape[1] if self.layers_count > 0 else self.shape[0]

        self.layers.append(D1FullLayer(neurons=neurons, inputs_per_neuron=layer_inputs, **kwargs))
        self.layers_count += 1

        self.shape = (self.shape[0], neurons)

    def calculate(self, x: np.array) -> np.array:
        if self.layers_count <= 0:
            raise LibraryException("Network is empty. Add more layers!")

        if len(x.shape) == 1 and x.shape[0] == self.shape[0]:
            result = x
            for layer in self.layers:
                result = layer.calculate(result)
            return result
        else:
            raise ShapeMismatchException("FullNetwork input must be ({},) shape, got {}".format(self.shape[0], x.shape))