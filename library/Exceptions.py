class LibraryException(Exception):
    def __init__(self, msg: str, errs: list = None):
        super().__init__(msg)
        self.errs = errs
        if self.errs:
            for e in self.errs:
                print("\t", e)


class DummyClassException(LibraryException):
    def __init__(self, classname: str):
        super().__init__("The {} class is unusable due to being a base class".format(classname))


class ShapeMismatchException(LibraryException):
    def __init__(self, msg: str):
        super().__init__(msg)