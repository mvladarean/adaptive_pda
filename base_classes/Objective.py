from abc import ABC, abstractmethod

class Objective(ABC):
    """
    Base Class for objective functions
    """
    def __init__(self):
        pass

    def func(self, x):
        pass

    def grad(self, x):
        pass

    def hessian(self, x):
        pass
