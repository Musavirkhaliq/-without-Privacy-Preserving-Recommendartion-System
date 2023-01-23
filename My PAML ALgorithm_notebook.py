#!/usr/bin/env python
# coding: utf-8

# In[231]:


def integrality_gap_elastic(e, kappa, lambda_):
    return min(kappa * abs(e) + lambda_ * e**2, kappa * abs(e - 1) + lambda_ * (e - 1)**2)


# In[232]:


def regularizer_elbmf(x, l1reg, l2reg):
    reg=0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            reg+=integrality_gap_elastic(x[i][j], l1reg, l2reg)
            
    return reg


# In[242]:


class ElasticBMF:
    def __init__(self,l1reg,l2reg):
        self.l1reg=l1reg
        self.l2reg=l2reg
f1 = ElasticBMF(0.1,0.2)
        


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


# In[241]:


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




