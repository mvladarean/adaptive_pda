from abc import ABC

class Regularizer(ABC):
    def __init__(self):
        pass

    def func(self, x):
        pass

    def prox(self, y, weight):
        pass

    def prox_of_conjugate(self, y, weight):
        if weight == 0:
            raise ValueError("weight in  prox is 0.")

        return y - weight * self.prox(y / weight, 1 / weight)

