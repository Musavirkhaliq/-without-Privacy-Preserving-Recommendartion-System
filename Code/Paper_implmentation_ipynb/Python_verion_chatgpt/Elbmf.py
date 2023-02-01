import random
import numpy as np
from scipy.sparse import csr_matrix

def integrality_gap_elastic(e, kappa, lambd):
    return min(kappa * abs(e) + lambd * e**2, kappa * abs(e - 1) + lambd * (e - 1)**2)

def regularizer_elbmf(x, l1reg, l2reg):
    return sum([integrality_gap_elastic(e, l1reg, l2reg) for e in x])

def proxel_1(x, k, l):
    return (x <= 0.5) * (x - k * np.sign(x)) + (x > 0.5) * (x - k * np.sign(x - 1) + l) / (1 + l)

def proxelp(x, k, l):
    return np.maximum(proxel_1(x, k, l), np.zeros_like(x))

def prox_elbmf(X, k, l):
    X = proxelp(X, k, l)

def proxelb(x, k, l):
    return np.clip(proxel_1(x, k, l), 0, 1)

def prox_elbmf_box(X, k, l):
    X = proxelb(X, k, l)

class ElasticBMF:
    def __init__(self, l1reg, l2reg):
        self.l1reg = l1reg
        self.l2reg = l2reg

class PALM:
    pass

class iPALM:
    def __init__(self, beta):
        self.beta = beta

def rounding(fn, args):
    for X in args:
        X = np.round(np.clip(prox_elbmf(X, 0.5, 1e20), 0, 1))

def apply_rate(fn, fn0, nu):
    fn.l2reg = fn0.l2reg * nu

def reducemf_impl(prox, opt, A, U, V, U_=None):
    VVt = V @ V.T
    AVt = A @ V.T
    L = max(np.linalg.norm(VVt), 1e-4)

    if opt == 'PALM':
        step_size = 1 / (1.1 * L)
        U = U - (U @ VVt - AVt) * step_size
        prox(U, step_size)
    elif isinstance(opt, iPALM):
        U = U + opt.beta * (U - U_)
        U_ = U
        step_size = 2 * (1 - opt.beta) / (1 + 2 * opt.beta) / L
        U = U - (U @ VVt - AVt) * step_size
        prox(U, step_size)
    return U

def reducemf(fn, opt, A, U, V, U_=None):
    if opt == 'PALM':
        return reducemf_impl(lambda x, alpha: prox_elbmf(x, fn.l1reg * alpha, fn.l2reg * alpha), opt, A, U, V)
    elif isinstance(opt, iPALM):
        return reducemf_impl(lambda x, alpha: prox_elbmf(x, fn.l1reg * alpha, fn.l2reg * alpha), opt, A, U, V, U_)

def factorize_palm(fn, X, U, V, regularization_rate, maxiter, tol, callback=None):
    ell = float('inf')
    for i in range(maxiter):
        reducemf(fn, 'PALM', X, U, V)
        reducemf(fn, iPALM(beta=0.5), X.T, V, U)
        ell_new = regularizer_elbmf(U @ V.T, fn.l1reg, fn.l2reg) + regularization_rate * np.linalg.norm(U) ** 2
        if callback:
            callback(i, ell, ell_new)
        if abs(ell - ell_new) < tol:
            break
        ell = ell_new
    return U, V

