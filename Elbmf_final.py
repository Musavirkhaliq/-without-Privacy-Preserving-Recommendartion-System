import numpy as np
import time
def integrality_gap_elastic(e, kappa, lambda_):
    return min(kappa * abs(e) + lambda_ * e**2, kappa * abs(e - 1) + lambda_ * (e - 1)**2)


def regularizer_elbmf(x, l1reg, l2reg):
    s=0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            s+=(integrality_gap_elastic(x[i][j], l1reg, l2reg))
    return s

def proxel_1(x, k, l):
    return (x - k * np.sign(x) if x <= 0.5 else (x - k * np.sign(x - 1) + l)) / (1 + l)

def proxelp(x, k, l):
    return np.maximum(proxel_1(x, k, l), np.zeros_like(x))

def prox_elbmf(X, k, l):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = proxelp(X[i][j], k, l)

def proxelb(x, k, l):
    return np.clip(proxel_1(x, k, l), 0, 1)

def prox_elbmf_box(X, k, l):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = proxelb(X[i][j], k, l)

class PALM:
    pass

class iPALM:
    def __init__(self, beta):
        self.beta = beta

class ElasticBMF:
    def __init__(self,l1reg,l2reg):
        self.l1reg=l1reg
        self.l2reg=l2reg

def rounding(fn, _, U: np.ndarray, V: np.ndarray) -> None:
    prox_elbmf(U, 0.5, 1e20)
    np.clip(U,0,1)
    np.round(U)
    
    prox_elbmf(V, 0.5, 1e20)
    np.clip(V,0,1)
    np.round(V)


def apply_rate(fn, fn0, nu):
    fn.l2reg = fn0.l2reg * nu


def gradient(A, U, V):
    VVt = V @ V.T
    AVt = A @ V.T
    return U @ VVt - AVt


"This computes the gradient of the objective function piece wise I need it to skip the unrated entries in the matrix A"
def recom_gradient(A, U, V):
    n,m=A.shape
    grad=[]
    for i in range(n):
        a=0
        for j in range(m):
            a += V[:,j]*((U[i]@V[:,j])-A[i][j])
        grad.append(a)
    grad=np.array(grad)
    return grad


def reducemf_impl(prox,p:PALM,A, U, V):
    VVt = V @ V.T
    # AVt = A @ V.T
    L = max(np.linalg.norm(VVt), 1e-4)
    step_size = 1 / (1.1 * L)
    # grad = U @ VVt - AVt
    grad = gradient(A, U, V)
    a,b = U.shape
    for i in range(a):
        for j in range(b):
            U[i][j] -= grad[i][j] * step_size
    prox(U, step_size)

def reducemf_impl_ip(prox,opt:iPALM,A, U, V,U_):
    VVt = V @ V.T
    # AVt = A @ V.T
    
    
    L = max(np.linalg.norm(VVt), 1e-4)
   
    
    a,b = U.shape
    
    #@. U = U + opt.beta * (U - U_)
    for i in range(a):
        for j in range(b):
            U[i][j]= U[i][j]+opt.beta*(U[i][j]-U_[i][j])
            
    #@. U_ = U
    for i in range(a):
        for j in range(b):
            U_[i][j] = U[i][j]
            
    step_size = 2*(1-opt.beta) / (1+2*opt.beta)/L
    
    # grad = U @ VVt - AVt
    grad = gradient(A, U, V)
    for i in range(a):
        for j in range(b):
            U[i][j] = U[i][j]-grad[i][j] * step_size
    prox(U, step_size)


def reducemf(fn, opt:PALM, A, U, V):
    def prox(x, alpha):
        return prox_elbmf(x, fn.l1reg * alpha, fn.l2reg * alpha)
    reducemf_impl(prox, opt, A, U, V)


def reducemf_i(fn, opt:iPALM, A, U, V,U_):
    def prox(x, alpha):
        return prox_elbmf(x, fn.l1reg * alpha, fn.l2reg * alpha)
    reducemf_impl_ip(prox, opt, A, U, V,U_)


import copy
def factorize_palm(fn, A, U, V, regularization_rate, maxiter, tol, callback=None):
    ell = float("inf")
    fn0 = copy.deepcopy(fn)
    for t in range(1,maxiter):
        fn.l2reg = fn0.l2reg * regularization_rate
        reducemf(fn, PALM,A, U, V)
        reducemf(fn, PALM,A.T, V.T, U.T)
#         ell, ell0 = np.linalg.norm(X - U @ V)**2, ell
#         ell0 = ell
#         ell = np.linalg.norm(X - U @ V)**2
#         if callback:
#             callback((U, V), ell)
#         if abs(ell - ell0) < tol:
#             break
    fn = fn0
    return U, V

def factorize_ipalm(
    fn,
    A,
    U,
    V,
    regularization_rate,
    maxiter,
    tol,
    beta,
    callback=None
):
    if beta == 0:
        return factorize_palm(fn, A, U, V, regularization_rate, maxiter, tol, callback=callback)

    ell = float("inf")
    fn0 = copy.deepcopy(fn)

    ip = iPALM(beta)
    U_ = np.copy(U)
    Vt_ = np.copy(V.T)

    for t in range(1,maxiter):
        fn.l2reg = fn0.l2reg *regularization_rate

        reducemf_i(fn, ip, A, U, V, U_)
        reducemf_i(fn, ip, A.T, V.T, U.T, Vt_)

#         ell0 = ell
#         ell = np.sum((A - np.dot(U, V)) ** 2)

#         if callback is not None:
#             callback((U, V), ell)
#         if abs(ell - ell0) < tol:
#             break
    fn = fn0
    return U, V

def init_factors(A,k):
    n,m=A.shape
    U=np.random.randint(2,size=(n, k)).astype("float")
    V=np.random.randint(2,size=(k, m)).astype("float")
    return U,V


def elbmf(
A,
k,
l1,
l2,
regularization_rate,
maxiter,
toleranace,
beta=0.0,
# batchsize=size(A,1),
roundi=True,
callback=None):
    n,m=A.shape
    U,V = init_factors(A,k)
    fn = ElasticBMF(l1,l2)
    U,V = factorize_ipalm(fn,A,U,V,regularization_rate,maxiter,toleranace,beta,callback=callback)
    roundi and rounding(fn,A,U,V)
    return U,V


def main():
    start_time = time.time()
    A = np.random.randint(2,size=(50, 60)).astype("float")
    k=10
    l1=0.01
    l2=0.02
    regularization_rate=1.3
    maxiter=5000
    tol=0.01
    beta=0.02
    U,V = elbmf(A,k,l1,l2,regularization_rate,maxiter,tol,beta)
    print("U = \n",U)
    print("V = \n",V)
    print("U.V = \n",U@V)
    print("A = \n",A)
    print("norm = \n",np.linalg.norm(A-U@V))
    end_time = time.time()
    print("Time taken: ", end_time - start_time, "seconds")

main()