import numpy as np

from library.Exceptions import DummyClassException


class Loss:
    @staticmethod
    def value(real: np.array, predicted: np.array) -> float:
        raise DummyClassException(Loss)

    @staticmethod
    def derive(real: np.array, predicted: np.array) -> float:
        raise DummyClassException(Loss)


class MeanSquaredError(Loss):
    @staticmethod
    def value(real: np.array, predicted: np.array) -> float:
        return ((real - predicted)**2).mean()

    @staticmethod
    def derive(real: np.array, predicted: np.array) -> float:
        return (-2*(real - predicted)).mean()
