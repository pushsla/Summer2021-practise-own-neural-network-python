import numpy as np

from library.Activation import SigmoidActivation
from library.Neuron import WeightedSumNeuron

a1 = SigmoidActivation()
n1 = WeightedSumNeuron(a1, 2, 1)

print(n1.calculate(np.array([1, 2])))
