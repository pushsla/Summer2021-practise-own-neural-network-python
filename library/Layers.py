import numpy as np

from library.Activations import Activation
from library.Exceptions import ShapeMismatchException, InternalShapeException


class D1FullLayer:
    def __init__(self, input_size: int, output_size: int, activation_type: type[Activation]):
        self.neural_shape = (input_size, output_size)
        self.matrix_shape = (output_size, input_size+1)
        self.inputs_shape = (input_size,)
        self.output_shape = (output_size,)
        self.neurons = np.random.rand(output_size, input_size+1)
        self.activation = activation_type

        self.last_unactivated = None
        self.last_neurons_unactivated = None

    def calculate(self, inputs: np.array) -> np.array:
        if self.inputs_shape != inputs.shape:
            raise ShapeMismatchException(D1FullLayer, self.calculate, self.inputs_shape, inputs.shape)

        output = np.dot(self.neurons, np.append(inputs, 1))
        if self.output_shape != output.shape:
            raise InternalShapeException(D1FullLayer, self.calculate, self.output_shape, output.shape)

        self.last_neurons_unactivated = output
        self.last_unactivated = np.sum(output)

        return np.array([self.activation.value(x) for x in output])


