from  base_classes.Optimizer import PrimalDualOptimizer, PrimalCompositeOptimizer
from  helpers import *
import math

import numpy as np
import time

class APDA(PrimalDualOptimizer):
    # Adaptive PDA for smooth objectives
    def __init__(self, x0, y0, objective, regularizer, num_iters, A, beta, is_strcnvx = False, as_operator=False, normA = None):
        super().__init__('APDA', x0, y0, objective, regularizer, num_iters, A)
        if beta <= 0:
            raise ValueError('beta should be strictly positive!')
        self.beta = beta
        self.is_strcnvx = is_strcnvx
        self.as_operator = as_operator
        self.ct = 1. / (1-1e-15)
        self.stepsizes = np.zeros((num_iters, 2))

        if normA != None:
            self.normA = normA
        elif as_operator == False:
            if scipy.sparse.issparse(A):
                self.normA = scipy.sparse.linalg.svds(A, k=1, return_singular_vectors=False)[0]
            else:
                self.normA = np.linalg.norm(A, 2)
        else:
            raise NotImplementedError


    def get_stepsizes(self, tau_k_prev, theta_k_prev, grad_k, grad_k_prev, x_k, x_k_prev):
        norm_grad = np.linalg.norm(grad_k - grad_k_prev, 2)
        norm_x = np.linalg.norm(x_k - x_k_prev, 2)

        if norm_grad == 0 and norm_x == 0:
            L_k = 0
        else:
            L_k = norm_grad / norm_x

        if self.is_strcnvx:
            inverse_lips = 0.5 * 1. / math.sqrt((2*L_k) ** 2 + self.beta * (self.normA ** 2))
            tau_growth = tau_k_prev * math.sqrt(1 + theta_k_prev / 2)
        else:
            inverse_lips = 0.5 * 1. / math.sqrt((L_k) ** 2 + self.ct * self.beta * (self.normA ** 2))
            tau_growth = tau_k_prev * math.sqrt(1 + theta_k_prev)

        tau_new = np.minimum(inverse_lips, tau_growth)
        sigma_new = self.beta * tau_new
        theta_new =  tau_new / tau_k_prev

        L_k2 = norm_grad / norm_x
        return tau_new, sigma_new, theta_new, L_k, L_k2

    def algorithm(self):
        y = self.y0.copy()
        x_prev = self.x0.copy()
        if self.as_operator == False:
            Ax_prev = self.A @ x_prev
        else:
            Ax_prev = self.A(x_prev)

        gradf_prev = self.objective.grad(x_prev)
        theta = 1
        tau = float('inf')

        x = x_prev - 1e-9 * (gradf_prev + self.A.T @ y) # take small step to get  an  x prev

        if self.as_operator == False:
            Ax = self.A @ x
        else:
            Ax = self.A(x)

        for i in range(self.num_iters):

            gradf = self.objective.grad(x)
            try:
                tau, sigma, theta, L_k, L_k2 = self.get_stepsizes(tau, theta, gradf, gradf_prev, x, x_prev)
            except:
                print("There was 0/0 division and we broke out of the algorithm to see what happens.\n The algo param is beta = " + str(self.beta) + "\n")
                break
            self.stepsizes[i][0] = tau
            self.stepsizes[i][1] = sigma

            Ax_bar = Ax + theta * (Ax - Ax_prev)
            y = self.regularizer.prox_of_conjugate(y + sigma * Ax_bar, sigma)

            x_prev, Ax_prev, gradf_prev = x, Ax, gradf

            if self.as_operator == False:
                x = x - tau * (gradf + self.A.T @ y)
                Ax = self.A @ x
            else:
                x = x - tau * (gradf + self.A.T(y))
                Ax = self.A(x)

            obj_val = self.objective.func(x)
            reg_val = self.regularizer.func(Ax)

            self.losses.append(obj_val + reg_val)

            if i % max(1, (self.num_iters//10)) == 0:
                self.print_progress(i, obj_val, reg_val, "stepsize tau, sigma = " + str(tau) + ", " + str(sigma))
                print("\n\n")

        return np.copy(self.losses), np.copy(x)

class CondatVu(PrimalDualOptimizer):
    def __init__(self, x0, y0, tau, sigma, objective, regularizer, num_iters, A):
        super().__init__('CVA', x0, y0, objective, regularizer, num_iters, A)
        self.tau = tau
        self.sigma = sigma

    def algorithm(self):
        y = self.y0.copy()
        x = self.x0.copy()
        Ax = self.A @ x
        failed = False

        for i in range(self.num_iters):
            x_next = x - self.tau * (self.objective.grad(x) + self.A.T @ y)
            Ax_next = self.A @ x_next

            Ax_bar = 2 * Ax_next - Ax
            y = self.regularizer.prox_of_conjugate(y + self.sigma * Ax_bar, self.sigma)

            x, Ax = x_next, Ax_next

            obj_val = self.objective.func(x_next)
            reg_val = self.regularizer.func(Ax_next)

            if math.isnan(obj_val) or math.isnan(reg_val)\
                    or math.isinf(obj_val) or math.isinf(reg_val):
                print("CVA failed with Nan or Inf. Stopping optimization.")
                failed = True
                break

            self.losses.append(obj_val + reg_val)

            if i % max(1, (self.num_iters//10)) == 0:
                self.print_progress(i, obj_val, reg_val,
                                    "stepsize tau, sigma = " + str(self.tau) + ", " + str(self.sigma))
                print("\n\n")

        if failed:
            return np.array([]), np.array([])

        return np.copy(self.losses), np.copy(x)

class FISTA(PrimalCompositeOptimizer):
    def __init__(self, x0, stepsize, objective, regularizer, num_iters, for_imaging=False):
        super().__init__('FISTA', x0, objective, regularizer, num_iters)
        self.stepsize = stepsize
        self.for_imaging = for_imaging

    def algorithm(self):
        return self.fista()

    def fista(self):
        x, y = self.x0.copy(), self.x0.copy()
        t = 1.

        num_iter_in_prox = 0
        x_opt = []
        for i in range(self.num_iters):
            if self.for_imaging:
                x1, num_iter = self.regularizer.prox(y - self.stepsize * self.objective.grad(y), self.stepsize)
                num_iter_in_prox += num_iter
            else:
                x1 = self.regularizer.prox(y - self.stepsize * self.objective.grad(y), self.stepsize)

            t1 = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            y = x1 + (t - 1) / t1 * (x1 - x)
            x, t = x1, t1

            obj_val = self.objective.func(y)
            reg_val = self.regularizer.func(y)

            if self.for_imaging:
                #print("for imaging2")
                self.losses.extend([obj_val + reg_val]*(num_iter))
                if num_iter_in_prox  >= self.num_iters:
                    x_opt = y.copy()
                    break
            else:
                self.losses.append(obj_val + reg_val)

            if i % max(1, (self.num_iters//10)) == 0:
                self.print_progress(i, obj_val, reg_val, "stepsize = " + str(self.stepsize))
                print("\n\n")

        print(num_iter_in_prox)

        if self.for_imaging:
            return np.copy(self.losses[:self.num_iters]), np.copy(x_opt)

        return np.copy(self.losses), np.copy(y)