import numpy as np
import matplotlib.pyplot as plt


def func(w,c,A,b):
    return np.mean(np.log(1 + np.exp(-b * (w.T.dot(A) + c))),axis=1)


def main():
    #Initialization
    m = 500
    n = 1000
    A = np.random.randn(n,m)
    b = np.sign(np.random.rand(1,m)-0.5)
    alpha = 0.1
    beta = 0.7
    epsilon = 1e-2

    w_k = np.ones([n, 1])
    c_k = np.ones([1,1])

    #Compute the gradient
    grad_w = np.mean((-b * np.exp(-b * (w_k.T.dot(A) + c_k))) / (1 + np.exp(-b * (w_k.T.dot(A) + c_k))) * A, axis=1).reshape([n,1])
    grad_c = np.mean((-b * np.exp(-b * (w_k.T.dot(A) + c_k))) / (1 + np.exp(-b * (w_k.T.dot(A) + c_k))), axis=1).reshape([1,1])

    norm = []
    norm.append(np.linalg.norm(np.vstack((grad_w,grad_c))))
    iter = 0
    while np.linalg.norm(np.vstack((grad_w,grad_c))) > epsilon:
        t = 1
        while func(w_k - t*grad_w, c_k - t*grad_c,A,b) > func(w_k,c_k,A,b)-alpha*t*np.linalg.norm(np.vstack((grad_w,grad_c))):
            t = beta*t
        w_k = w_k - t*grad_w
        c_k = c_k - t*grad_c
        print(np.linalg.norm(np.vstack((grad_w,grad_c))))
        grad_w = np.mean((-b * np.exp(-b * (w_k.T.dot(A) + c_k))) / (1 + np.exp(-b * (w_k.T.dot(A) + c_k))) * A, axis=1).reshape([n,1])
        grad_c = np.mean((-b * np.exp(-b * (w_k.T.dot(A) + c_k))) / (1 + np.exp(-b * (w_k.T.dot(A) + c_k))), axis=1).reshape([1,1])
        norm.append(np.linalg.norm(np.vstack((grad_w, grad_c))))
        iter += 1


    plt.figure()
    plt.plot(norm)
    plt.show()

if __name__ == "__main__":
    main()