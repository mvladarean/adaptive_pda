from  base_classes.Regularizer import Regularizer
import time
import math
from helpers import TV_norm
from skimage.restoration import denoise_tv_chambolle
from scipy.sparse import spdiags, kron, identity
import scipy
import numpy as np

class L1Norm(Regularizer):
    def __init__(self, lmbd):
        # prox of lmbd* || x ||_1
        self.lmbd = lmbd
        super().__init__()

    def func(self, x):
        return self.lmbd * np.linalg.norm(x, 1)

    def prox(self, y, weight):
        if weight == 0:
            raise ValueError("weight in  prox is 0.")

        return y + np.clip(-y, -weight * self.lmbd, weight * self.lmbd)

class QuadraticEuclidean(Regularizer):
    # f(x) = 1 / 2 * | | x - b | | ^ 2
    def __init__(self, b):
        super().__init__()
        self.b = b

    def func(self, x):
        return 0.5 * np.linalg.norm(x - self.b)**2

    def prox(self, y, weight):
        return (y + weight * self.b) / (1 + weight)

class TVNormForPrimal(Regularizer):
    def __init__(self, lmbd, is_iso = False):
        super().__init__()
        self.lmbd = lmbd
        self.is_iso = is_iso

    def func(self, x):
        # receives vector, reshapes into mxm image where m is sqrt of first dimension
        size_img = math.sqrt(x.shape[0])
        assert size_img == int(size_img)

        return self.lmbd * TV_norm(x.reshape((int(size_img), int(size_img))), self.is_iso)

    def prox(self, y, weight):
        # receives vector, reshapes into mxm image where m is sqrt of first dimension
        img_vec_len = y.shape[0]
        two_dim_img_size = math.sqrt(img_vec_len)
        assert two_dim_img_size == int(two_dim_img_size)

        y, iter =  denoise_tv_chambolle(y.reshape((int(two_dim_img_size), int(two_dim_img_size))),
                                                  weight=self.lmbd * weight, eps=1e-5,
                                                  n_iter_max=100)

        return y.reshape((img_vec_len, 1)), iter


class TVNormForPD(Regularizer):
    def __init__(self, lmbd, is_iso = False):
        super().__init__()
        self.lmbd = lmbd
        self.is_iso = is_iso
        if is_iso:
            p = 2
        else:
            p = 1

    def func(self, x):
        # Receives a vector of size 2N where N =  is the vectorized application of the x gradient
        # and 2nd N is the vectorized app of the y gradient. I.e. Receives as input D@x, where x is the vectorized image
        # and D is [Dx; Dy]
        len_x = len(x) // 2
        if not self.is_iso:
            return self.lmbd * np.sum(np.abs(x))

        z = [math.sqrt(x[i] ** 2 + x[len_x + i] ** 2) for i in range(len_x)]
        return self.lmbd * np.sum(z)

    def prox(self, y, weight):
        # receives vector, reshapes into mxm image where m is sqrt of first dimension
        raise NotImplementedError("The direct prox is not implementable, needs iterative method and not used in PD algs.")

    def prox_of_conjugate(self, y, weight):
        # The conjugate function is \delta_{||.||_{p, infty} <= lmbd}(y), where y in R^{m x m x 2}. Because of representation problems in python
        # The vector y is  recoded as size (2m^2, 1), where the first half corresponds to the Dx difference operator andthe second to the Dy
        # The prox is therefore a projection onto the the polar norm  ball of radius q -> infty of radius lmbd, where q is such that 1/p + 1/q = 1.
        # projection formula taken from chambolle pock's an introduction to optimization for imaging eq 4.23
        result = np.zeros_like(y)
        lny = len(y)
        assert lny // 2 == lny / 2
        true_y_len = lny // 2
        if self.is_iso:
            for i in range(true_y_len):
                # TODO can be optimized instead of for loop
                denominator = np.maximum(1., math.sqrt(y[i]**2 + y[true_y_len + i]**2)/(self.lmbd))
                result[i] = y[i]/denominator
                result[true_y_len + i] = y[true_y_len + i]/denominator

            return result
        else:
            raise NotImplementedError


    def build_finite_diff_operator(self, size_img, as_operator = False):
       if as_operator == False:
           return self.build_finite_diff_matrix(size_img)
       else:
           return self.build_finite_diff_linop(size_img)

    def build_finite_diff_matrix(self, size_img):
        # implementation adapted from:
        # https://regularize.wordpress.com/2013/06/19/how-fast-can-you-calculate-the-gradient-of-an-image-in-matlab/
        # Assumes squared size image for simplicity, and size_img represents m^2
        M = int(math.sqrt(size_img))
        assert M == math.sqrt(size_img)  # check that it is proper square

        first_diag = (-1) * np.ones(M)
        second_diag = np.ones(M)
        data = np.array([first_diag, second_diag])
        diags = np.array([0, 1])
        diag_mat = spdiags(data, diags, M, M, format="csr")
        diag_mat[M - 1, M - 1] = 0
        Dx = kron(diag_mat, identity(M, format="csr"))
        Dy = kron(identity(M, format="csr"), diag_mat)

        return scipy.sparse.vstack([Dy, Dx])

    def build_finite_diff_linop(self, size_img):
        # Assumes squared size image for simplicity, and size_img represents m^2
        M = int(math.sqrt(size_img))
        assert M == math.sqrt(size_img)  # check that it is proper square

        return FiniteDiffOperator(M)


class FiniteDiffOperator():
    def __init__(self, M):
        # assumes image is MxM
        self.M = M

        first_diag = (-1) * np.ones(M)
        second_diag = np.ones(M)
        data = np.array([first_diag, second_diag])
        diags = np.array([0, 1])
        self.diff_mat = spdiags(data, diags, M, M, format="csr")
        self.diff_mat[M - 1, M - 1] = 0

    def grad(self, x):
        # receives x of size M^2
        x = np.reshape(x, (self.M, self.M))

        Dxu = self.diff_mat @ x
        Dyu = x @ self.diff_mat.T

        return np.vstack((Dyu.reshape((self.M**2, 1)), Dxu.reshape((self.M**2, 1))))

    def div(self, y):
        # receives x of size M^2
        assert self.M**2 == len(y)//2

        y1 = y[:self.M**2].reshape(self.M, self.M)
        y2 = y[self.M**2:].reshape(self.M, self.M)

        x1 = y1 @ self.diff_mat
        x2 = self.diff_mat.T @ y2

        x = (x1 + x2).reshape((self.M**2, 1))

        return x

    def get_norm(self):
        return np.sqrt(8)