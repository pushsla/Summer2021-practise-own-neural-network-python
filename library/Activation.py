from library.Exceptions import DummyClassException
from numpy import exp


class Activation:
    def __init__(self) -> None:
        raise DummyClassException("Activation.Activation")

    def value(self, x: float) -> float:
        return x

    def derivative(self, x: float) -> float:
        return 0


class SigmoidActivation(Activation):
    def __init__(self) -> None:
        pass

    def value(self, x: float) -> float:
        ex = exp(x)
        return ex/(ex+1)

    def derivative(self, x: float) -> float:
        val = self.value(x)
        return val*(1-val)
