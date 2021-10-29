from  base_classes.Objective import Objective
from  helpers import *
import  time
import scipy
import scipy.special
import scipy.sparse.linalg as spr_LA

import numpy as np

class LeastSquares(Objective):
    def __init__(self, A, b):
        super().__init__()

        self.A = A
        self.b = b
        if scipy.sparse.issparse(A):
            self.normA = scipy.sparse.linalg.svds(A, k=1, return_singular_vectors=False)[0]
        else:
            self.normA = np.linalg.norm(A, 2)
        self.Atb = A.T @ b
        self.AtA = A.T @ A

    def func(self, x):
        return 0.5 * np.linalg.norm(self.A @ x - self.b, 2)**2

    def grad(self, x):
        return self.AtA @ x - self.Atb

    def lips(self):
        return self.normA**2

class PhaseRetrieval(Objective):
    def __init__(self, A, b):
        super().__init__()

        self.A = A
        self.b = b
        self.m = A.shape[0]

    def func(self, x):
        return 0.25/self.m * np.sum(np.square(np.square(self.A@x) - self.b))

    def grad(self, x):
        Ax = self.A @ x
        tmp = Ax ** 3 - np.multiply(self.b, Ax)

        return (self.A.T @ tmp)/self.m

    def lips(self):
        raise ValueError("There is no global lipschitz constant for the phase retrieval objective.")

class LogisticRegression(Objective):
    def __init__(self, A, b):
        super().__init__()
        assert b.shape[1] == 1  # column vector
        assert b.shape[0] == A.shape[0]  # num of entries in b should match rows in A

        if scipy.sparse.issparse(A):
            self.K = A.multiply(-b)
            self.normA = scipy.sparse.linalg.svds(self.K, k=1, return_singular_vectors=False)[0]
        else:
            self.K = np.multiply(-b, A)
            self.normA = np.linalg.norm(self.K, 2)
        self.A = A
        self.b = b

    def func(self, x):
        # return sum_{i = 1}^m log(1 + exp(-b_i *< a_i, x>))
        assert x.shape[1] == 1
        assert self.K.shape[1] == x.shape[0]
        Kx = self.K @ x # gives vector

        return np.sum(np.log(1. + np.exp(Kx)))

    def grad(self, x):
        # return sum_{i = 1}^m \frac{-b_i * a_i}{ 1 + exp(b_i< a_i, x>)}

        assert x.shape[1] == 1  # column vector
        assert self.K.shape[1] == x.shape[0]

        Kx = self.K @ x
        tmp = scipy.special.expit(Kx) # better  for precision errors
        return self.K.T @ tmp

    def hessian(self, x):
        Kx = self.K @ x
        d = np.multiply(scipy.special.expit(Kx), 1 - scipy.special.expit(Kx))
        D = scipy.sparse.diags(d, [0], format="csr")

        return self.A.T @ D @ self.A

    def lips(self):
        return 0.25 * (self.normA **2)
