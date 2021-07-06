class LibraryException(Exception):
    def __init__(self, msg: str, errs: list = None):
        super().__init__(msg)
        self.errs = errs
        if self.errs:
            for e in self.errs:
                print("\t", e)


class DummyClassException(LibraryException):
    def __init__(self, classtype: type):
        super().__init__("The {} class is unusable due to being a base class".format(classtype.__name__))


class ShapeMismatchException(LibraryException):
    def __init__(self, classtype: type, function, need: tuple[int], got: tuple[int]):
        super().__init__("{} expected {} shape in function {}, got {}".format(classtype.__name__, need, function.__name__, got))


class InternalShapeException(ShapeMismatchException):
    pass
