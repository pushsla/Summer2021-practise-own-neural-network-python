import numpy as np

from library.DeprecatedExceptions import DummyClassException


class Loss:
    def __init__(self):
        raise DummyClassException("Loss.Loss")

    def value(self, true: np.array, predicted: np.array) -> float:
        pass

    def derivative(self, true: np.array, predicted: np.array) -> float:
        pass


class LossMSE(Loss):
    def __init__(self):
        pass

    def value(self, true: np.array, predicted: np.array) -> float:
        return ((true - predicted) ** 2).mean()

    def derivative(self, true: np.array, predicted: np.array) -> float:
        return (-2*(true-predicted)).mean()