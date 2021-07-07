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
        errors = np.array([((real[i] - predicted[i])**2).sum() for i, _ in enumerate(real)])
        return errors.mean()

    @staticmethod
    def derive(real: np.array, predicted: np.array) -> float:
        errors = np.array([(-2*(real[i] - predicted[i])).sum() for i, _ in enumerate(real)])
        return errors.mean()
