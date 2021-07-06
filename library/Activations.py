from math import exp

from library.Exceptions import DummyClassException


class Activation:
    @staticmethod
    def value(x: float) -> float:
        raise DummyClassException(Activation)

    @staticmethod
    def derive(x: float) -> float:
        raise DummyClassException(Activation)


class SigmoidActivation(Activation):
    @staticmethod
    def value(x: float) -> float:
        ex = exp(x)
        return ex/(ex+1)

    @staticmethod
    def derive(x: float) -> float:
        val = SigmoidActivation.value(x)
        return val*(1-val)
