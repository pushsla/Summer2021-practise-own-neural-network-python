from library.DeprecatedExceptions import ShapeMismatchException, LibraryException
from library.DeprecatedLayer import D1FullLayer
import numpy as np

from library.DeprecatedLoss import Loss, LossMSE


class D1FullNetwork:
    def __init__(self, input_size: int, loss_function_type: type[Loss] = LossMSE) -> None:
        self.shape = (input_size,)
        self.layers_count = 0
        self.layers = []

        self.loss_function_type = loss_function_type
        self.loss_function = loss_function_type()

    def add_layer(self, neurons: int, **kwargs):
        layer_inputs = self.layers[-1].shape[1] if self.layers_count > 0 else self.shape[0]

        self.layers.append(D1FullLayer(neurons=neurons, inputs_per_neuron=layer_inputs, **kwargs))
        self.layers_count += 1

        self.shape = (self.shape[0], neurons)

    def fit(self, x: np.array, y: np.array):
        if x.shape[1] == self.shape[0] and y.shape[1] == self.shape[1]:
            for i, xdata in enumerate(x):
                pass
        else:
            raise ShapeMismatchException("Excepted X: ({},*) and Y: ({},*). Got X: {}, Y: {}".format(self.shape[0], self.shape[1], x.shape, y.shape))


    def calculate(self, x: np.array) -> np.array:
        if self.layers_count <= 0:
            raise LibraryException("Network is empty. Add more layers!")

        if len(x.shape) == 1 and x.shape[0] == self.shape[0]:
            result = [x]
            for layer in self.layers:
                result.append(layer.calculate(result[-1]))
            return np.array(result)
        else:
            raise ShapeMismatchException("FullNetwork input must be ({},) shape, got {}".format(self.shape[0], x.shape))