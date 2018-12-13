#Subgradient method to solve Lasso Problem
#Minhui Huang 12/13/2018

#ISTA algorithm

import numpy as np
#import matplotlib.pyplot as plt

def compute_grad(x_k,A,b):
    grad_l2 = A.T.dot(A.dot(x_k)-b)
    grad = grad_l2
    return grad

def f(x_k,A,b):

    return 0.5*np.linalg.norm(A.dot(x_k)-b)**2

def model(x,xk,A,b,GammaK):

    innerProd = compute_grad(xk,A,b).T.dot(x - xk)
    xDiff = x - xk
    return f(xk,A,b) + innerProd + (1.0/(2.0*GammaK))*xDiff.T.dot(xDiff)

def main():

    # initialization
    m = 100
    n = 500
    s = 5
    A = np.random.randn(m, n)
    xs = np.zeros([n, 1])
    picks = np.random.permutation(np.arange(1, n))
    xs[picks[0:s], 0] = np.random.randn(s, 1).flatten()
    b = A.dot(xs)

    x_k = np.zeros([n, 1])
    epsilon = 1e-2
    tau = 1
    t = 0.1
    beta = 0.7

    k = 1
    error = []

    while np.linalg.norm(x_k - xs) / np.linalg.norm(xs) >= epsilon:
        while f(x_k - t*compute_grad(x_k,A,b),A,b) > model(x_k - t*compute_grad(x_k,A,b),x_k,A,b,t):
            t = beta*t
        u = x_k - t*compute_grad(x_k,A,b)
        x_k = np.sign(u)*np.maximum(np.abs(u)-tau, np.zeros(np.shape(u)))
        k += 1
        error.append(np.linalg.norm(x_k - xs) / np.linalg.norm(xs))
        print(np.linalg.norm(x_k - xs))
        print(np.linalg.norm(xs))

if __name__ == "__main__":
    main()