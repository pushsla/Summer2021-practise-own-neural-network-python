class LibraryException(Exception):
    def __init__(self, msg: str, errs: list = None):
        super().__init__(msg)
        self.errs = errs
        for e in self.errs:
            print("\t", e)


class DummyClassException(LibraryException):
    def __init__(self, classname: str):
        super().__init__("The {} class is unusable".format(classname))

