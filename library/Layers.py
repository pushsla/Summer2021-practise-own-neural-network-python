import numpy as np

from library.Activations import Activation
from library.Exceptions import ShapeMismatchException, InternalShapeException, DummyClassException


class Layer:
    def __init__(self, input_size: int, output_size: int, activation_type: type[Activation]):
        self.neural_shape = (input_size, output_size)
        self.matrix_shape = (output_size, input_size+1)
        self.inputs_shape = (input_size,)
        self.output_shape = (output_size,)
        self.activation = activation_type
        self.neurons = None

        self.last_inputs = None
        self.last_activated = None
        self.last_unactivated = None
        self.last_neurons_unactivated = None

    def calculate(self, inputs: np.array) -> np.array:
        raise DummyClassException(Layer)


class InputLayer(Layer):
    def __init__(self, input_size: int):
        super().__init__(input_size, input_size, Activation)

    def calculate(self, inputs: np.array) -> np.array:
        self.last_inputs = np.ndarray.copy(inputs)
        self.last_neurons_unactivated = np.ndarray.copy(inputs)
        self.last_unactivated = np.sum(inputs)
        self.last_activated = np.ndarray.copy(inputs)

        return self.last_activated


class D1FullLayer(Layer):
    def __init__(self, input_size: int, output_size: int, activation_type: type[Activation]):
        super().__init__(input_size, output_size, activation_type)

        self.neurons = np.random.rand(output_size, input_size+1)

    def calculate(self, inputs: np.array) -> np.array:
        if self.inputs_shape != inputs.shape:
            raise ShapeMismatchException(D1FullLayer, self.calculate, self.inputs_shape, inputs.shape)

        self.last_inputs = np.append(inputs, 1)
        output = np.dot(self.neurons, self.last_inputs)
        if self.output_shape != output.shape:
            raise InternalShapeException(D1FullLayer, self.calculate, self.output_shape, output.shape)

        self.last_activated = np.array([self.activation.value(x) for x in output])
        self.last_neurons_unactivated = output
        self.last_unactivated = np.sum(output)

        return self.last_activated


