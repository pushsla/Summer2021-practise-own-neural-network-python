import numpy as np

from library.Activations import Activation
from library.Layers import Layer, InputLayer
from library.Losses import Loss


class Network:
    def __init__(self, input_size: int):
        self.inputs_shape = (input_size,)
        self.output_shape = (None,)
        self.layers: list[Layer] = [InputLayer(self.inputs_shape[0])]

    def add(self, layer_type: type[Layer], output_size: int, activation_type: type[Activation]):
        input_size = self.layers[-1].output_shape[0]

        self.layers.append(layer_type(input_size=input_size, output_size=output_size, activation_type=activation_type))
        self.output_shape = self.layers[-1].output_shape

    def predict_many(self, x: np.array) -> np.array:
        result = []
        for example_num, example in enumerate(x):
            result.append(self.predict(example))

        return np.array(result)

    def predict(self, x: np.array) -> np.array:
        output = np.ndarray.copy(x)
        for layer in self.layers:
            output = layer.calculate(output)

        return output

    def fit(self, x: np.array, y: np.array, loss_type: type[Loss], lr: float = 0.1):
        predicted = self.predict_many(x)
        loss = loss_type.value(x, predicted)
        dL_dPred = loss_type.derive(y, predicted)

        for layer_num, _ in reversed(tuple(enumerate(self.layers))[1:-1]):
            next_l = self.layers[layer_num+1]
            layer = self.layers[layer_num]

            for neuron_num, neuron in enumerate(layer.neurons):
                dPred_dH = (next_l.neurons[:, neuron_num]*[next_l.activation.derive(x) for x in next_l.last_unactivated]).mean()

                for w_num, _ in enumerate(neuron):
                    p1 = layer.last_inputs[w_num]
                    p2 = layer.activation.derive(layer.last_unactivated[neuron_num])
                    dH1_dW = p1*p2
                    layer.neurons[neuron_num][w_num] -= dL_dPred*dPred_dH*dH1_dW*lr
