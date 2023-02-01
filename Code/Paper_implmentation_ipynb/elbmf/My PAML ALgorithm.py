#!/usr/bin/env python
# coding: utf-8

# In[709]:


A = [[1, 0, 0, 1, 0],[1, 0, 1, 0, 1], [0, 1, 1, 0, 1], [0, 1, 0, 0, 1], [1, 1, 1, 0, 1]]
U = [[1.9 ,2.5 ,1.05], [2.2, 4.2, 0.9], [4.1, 0.36, 3.1], [5.0, 3.1, 2.36],[1.2, 7.002, 6.3]]
V = [[1.2, 6.21, 1.09, 8.01, 2.36], [2.1, 7.0, 6.36, 0.63, 1.01], [1.002, 9, 6.2, 5.8, 0.05]]
X=[[2,0.63,1.2,1.2],[2.1,0.1,0.6,0.3],[6.3,1.2,5.6,0.19]]
print("A=",A)
print("U=",U)
print("V=",V)
print("X=",X)


# In[702]:


import numpy as np
A=np.array(A)
U=np.array(U)
V=np.array(V)
X=np.array(X)
print("A=\n",A)
print("U=\n",U)
print("V=\n",V)
print("X=\n",X)


# # integrality_gap_elastic

# In[672]:


def integrality_gap_elastic(e, kappa, lambda_):
    return min(kappa * abs(e) + lambda_ * e**2, kappa * abs(e - 1) + lambda_ * (e - 1)**2)


# In[673]:


integrality_gap_elastic(2,0.01,0.02)


# # regularizer_elbmf

# In[674]:


def regularizer_elbmf(x, l1reg, l2reg):
    s=0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            s+=(integrality_gap_elastic(x[i][j], l1reg, l2reg))
    return s


# In[675]:


regularizer_elbmf(X,0.001,0.002)


# # proxel_1

# In[676]:


def proxel_1(x, k, l):
    return (x - k * np.sign(x) if x <= 0.5 else (x - k * np.sign(x - 1) + l)) / (1 + l)


# In[677]:


proxel_1(4,0.1,0.3)


# # proxelp

# In[678]:


def proxelp(x, k, l):
    return np.maximum(proxel_1(x, k, l), np.zeros_like(x))


# In[679]:


proxelp(3, 0.1, 0.2)


# # prox_elbmf

# In[680]:


def prox_elbmf(X, k, l):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = proxelp(X[i][j], k, l)


# In[681]:


prox_elbmf(X, 0.01, 0.02)


# In[682]:


print(X)


# # proxelb

# In[683]:


def proxelb(x, k, l):
    return np.clip(proxel_1(x, k, l), 0, 1)


# In[684]:


proxelb(2,0.01,0.02)


# # prox_elbmf_box

# In[685]:


def prox_elbmf_box(X, k, l):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = proxelb(X[i][j], k, l)


# In[686]:


prox_elbmf_box(X,0.0,0.02)


# In[687]:


print(X)


# In[ ]:





# # structs

# In[688]:


class PALM:
    pass

class iPALM:
    def __init__(self, beta):
        self.beta = beta


# In[689]:


class ElasticBMF:
    def __init__(self,l1reg,l2reg):
        self.l1reg=l1reg
        self.l2reg=l2reg


# In[690]:


fn = ElasticBMF(0.001,0.002)


# # rounding

# In[691]:


def rounding(fn, _, U: np.ndarray, V: np.ndarray) -> None:
    prox_elbmf(U, 0.5, 1e20)
    np.clip(U,0,1)
    np.round(U)
    
    prox_elbmf(V, 0.5, 1e20)
    np.clip(V,0,1)
    np.round(V)
    


# In[692]:


U


# In[693]:


V


# In[694]:


rounding(fn,A,U,V)


# In[695]:


U


# In[696]:


V


# # apply rate

# In[697]:


def apply_rate(fn, fn0, nu):
    fn.l2reg = fn0.l2reg * nu


# In[698]:


apply_rate(fn,fn,2)


# In[699]:


fn.l1reg


# In[700]:


fn.l2reg


# # PALM reducemf_impl

# In[715]:


def reducemf_impl( A, U, V):
    VVt = V @ V.T
    AVt = A @ V.T
    L = max(np.linalg.norm(VVt), 1e-4)
    step_size = 1 / (1.1 * L)
    grad = U @ VVt - AVt
#     grad=0
    U -= grad * step_size
#     prox(U, step_size)


# In[716]:


A = np.random.randint(2, size=(100,400))
# R=np.array([
#    [1,0,0,4,5,0],
#    [0,0,3,3,0,4],
#    [0,4,0,2,0,3],
#    [2,0,5,0,1,3],
m,n = A.shape
#   ])
K = 3
U = np.random.randint(2, size=(n,k))
V = np.random.randint(2, size=(k,m))

print("A=\n",A)
print("U=\n",U)
print("V=\n",V)
print("UV=\n",np.dot(U,V))





nU, nV = reducemf_impl(A,U,V)

nR = np.dot(nU, nR)
print(nR)
print("")
print(np.linalg.norm(R-nR)**2)


# In[ ]:





# In[ ]:





# # reduce emf!

# In[634]:


def reducemf(fn, PALM, A, U, V):
    def prox(x, alpha):
        return prox_elbmf(x, fn.l1reg * alpha, fn.l2reg * alpha)
    reducemf_impl(prox, PALM, A, U, V)


# In[635]:


reducemf(fn,PALM,A,U,V)


# In[637]:


A.shape


# In[572]:


U


# In[573]:


V


# In[ ]:





# # factorize_palm

# In[655]:


import copy
def factorize_palm(fn, X, U, V, regularization_rate, maxiter, tol, callback=None):
    ell = float("inf")
    fn0 = copy.deepcopy(fn)
    for t in range(1,maxiter):
        fn.l2reg = fn0.l2reg * regularization_rate**t
        reducemf(fn, PALM,X, U, V)
        reducemf(fn, PALM,X.T, V.T, U.T)
#         ell, ell0 = np.linalg.norm(X - U @ V)**2, ell
        if callback:
            callback((U, V), ell)
#         if abs(ell - ell0) < tol:
#             break
    fn = fn0
    return U, V


# # calling function

# In[668]:


# A = np.random.randint(2,size=(300, 400))
# print(A)


# In[708]:


# def factorize(A,k,l1reg,l2reg,regularization_rate,tol):



n,m=A.shape
print(n,m)
k=4
# U = np.random.randint(2,n, k)
# V = np.random.randint(2,k, m)
U,V=factorize_palm(fn,A,U,V,0.1,1000,0.01)
# rounding(fn,A,U,V)
A.shape


# In[707]:


print("A=\n",A)
print("U=\n",U)
print("V=\n",V)
print("UV=\n",np.dot(U,V))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[243]:





# In[238]:


import numpy as np

def matrix_factorization_PALM(fn,R, k, lambda_1=0.01,lambda_2=0.02, max_iter=1000, tol=1e-4):
    """
    Matrix factorization using Proximal Alternating Linear Minimization (PALM)
    
    Parameters:
        R (np.ndarray): Input matrix to factorize
        k (int): Number of latent factors
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance for convergence
        lambda_ (float): Regularization term
        
    Returns:
        (np.ndarray, np.ndarray): The two factor matrices
    """
    
    m, n = R.shape
    X = np.random.rand(m, k)
    Y = np.random.rand(k, n)
    
    # Set the step size
    eta = 1 / (np.linalg.norm(X, 2) * np.linalg.norm(Y, 2))
    fn0=copy.deepcopy(fn)
    for i in range(max_iter):
        fn.l2reg = fn0.l2reg * regularization_rate(t - 1)
#         print(i)
    #         X_prev = X.copy()
    #         Y_prev = Y.copy()

        # Minimize with respect to X

#         for j in range(m):
        YYt = Y @ Y.T
        RYt = R @ Y.T
        L = max(np.linalg.norm(YYt), 1e-4)
        eta = 1/(1.1*L)
        reg=regularizer_elbmf(X,lambda_1,lambda_2)
        grad_X = X @ YYt - RYt +reg * np.sign(X)
#         print(grad_X)
#     * np.sign(R)
#         +reg * np.sign(X)
        X = X-grad_X * eta


        # Minimize with respect to Y
#         for j in range(n):
        XtX = X.T @ X
        RtX = R.T @ X
        L = max(np.linalg.norm(XtX), 1e-4)
        eta = 1/(1.1*L)
        reg=regularizer_elbmf(Y.T,lambda_1,lambda_2)
        grad_Yt = Y.T @ XtX - RtX +reg * np.sign(Y.T)
#     * np.sign(R)
#         +reg * np.sign(Y.T)
        Yt = Y.T-grad_Yt *eta


        
#         ell = np.linalg.norm(R - X @ Y)**2
#         ell0 = ell
        # Check for convergence
    #         if (abs(ell - ell0) < tol):
    #             break   
    return X, Y


# In[714]:


R = np.random.randint(6, size=(100,400), dtype=np.int8)
# R=np.array([
#    [1,0,0,4,5,0],
#    [0,0,3,3,0,4],
#    [0,4,0,2,0,3],
#    [2,1,5,0,1,3],
#   ])


print(R)
print("")


K = 10


nP, nQ = matrix_factorization_PALM(R,K)

nR = np.dot(nP, nQ)
print(nR)
print("")
print(np.linalg.norm(R-nR)**2)


# # Dataset

# In[220]:


# Reads the movie lens data and creates a matrix out of it 

import numpy as np
import pandas as pd
ratings = pd.read_csv("/home/cris-musa/Documents/Dataset/ratings.csv")
movies = pd.read_csv("/home/cris-musa/Documents/Dataset/movies.csv")
# print(ratings)
# print(movies)

R = np.zeros(shape=(610,193609),dtype=float)
for i in range(100835):
    user = ratings["userId"][i]-1
    movie = ratings["movieId"][i]-1
    R[user][movie]=ratings["rating"][i]
print(R)
print(np.sign(R))


# In[221]:


print(R)
nP, nQ = matrix_factorization_PALM(R,40)
nR = np.dot(nP, nQ)
print(nR/5)
print("")
print(np.linalg.norm(R-nR)**2)


# In[229]:


# R = np.random.randint(6, size=(100,400), dtype=np.int8)
R=np.array([
   [1,0,0,4,5,0],
   [0,0,3,3,0,4],
   [0,4,0,2,0,3],
   [2,1,5,0,1,3],
  ])


print(R)
print("")


K = 3


nP, nQ = matrix_factorization_PALM(R,K)

nR = np.dot(nP, nQ)
print(nR)
print("")
print(np.linalg.norm(R-nR)**2)


# In[ ]:





# In[183]:





# In[217]:


A = np.array([
    [1,2,3,4],
    [0,1,0,5],
    [4,0,1,3],
])
B = np.array([
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1],
])

A*5


# In[224]:


import numpy as np

def ALS(A, k, max_iter=100, tol=1e-4):
    m, n = A.shape
    U = np.random.rand(m, k)
    V = np.random.rand(n, k)
    for t in range(max_iter):
        # Update U
        V_T_V = np.dot(V.T, V)
        A_V = np.dot(A, V)
        for i in range(m):
            u_i = np.dot(A_V[i, :], np.linalg.inv(V_T_V))
            U[i, :] = u_i / np.linalg.norm(u_i)
        
        # Update V
        U_T_U = np.dot(U.T, U)
        A_U = np.dot(A.T, U)
        for j in range(n):
            v_j = np.dot(A_U[j, :], np.linalg.inv(U_T_U))
            V[j, :] = v_j / np.linalg.norm(v_j)
        
        # Check for convergence
        if np.linalg.norm(A - np.dot(U, V.T)) < tol:
            break
    return U, V


# In[227]:


# R = np.random.randint(6, size=(100,400), dtype=np.int8)
R=np.array([
   [1,0,0,4,5,0],
   [0,0,3,3,0,4],
   [0,4,0,2,0,3],
   [2,0,5,0,1,3],
  ])


print(R)
print("")


K = 3


nP, nQ = ALS(R,K)

nR = np.dot(nP, nQ.T)
print(nR)
print("")
print(np.linalg.norm(R-nR)**2)


# In[ ]:




