from abc import ABC
import matplotlib.pyplot as plt

class PrimalCompositeOptimizer(ABC):
    def __init__(self, label, x0, objective, regularizer, num_iters):
        self.label = label
        self.x0 = x0.copy()
        self.objective = objective
        self.regularizer = regularizer
        self.num_iters = num_iters
        self.losses = []
        self.has_run = False

    def run(self):
        self.has_run = True
        self.losses = [self.compute_loss(self.x0), self.compute_loss(self.x0)]
        print("\n\n========================= START - " + self.label + " =========================\n")

        return self.algorithm()

    def algorithm(self):
        pass

    def compute_loss(self, x):
        return  self.objective.func(x) + self.regularizer.func(x)

    def print_progress(self, num_iter, obj_val, reg_val, stepsize = None):
        print("----iter = " + str(num_iter))
        print("----obj = " + str(obj_val))
        print("----feas = " + str(reg_val))
        if stepsize !=  None:
            print("----" + str(stepsize))



class PrimalDualOptimizer(PrimalCompositeOptimizer):
    def __init__(self, label, x0, y0, objective, regularizer, num_iters, A):
        super().__init__(label, x0, objective, regularizer, num_iters)
        self.y0 = y0.copy()
        self.A = A

    def compute_loss(self, x):
        return  self.objective.func(x) + self.regularizer.func(self.A @ x)

