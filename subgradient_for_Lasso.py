import numpy as np
#import matplotlib.pyplot as plt

#Compute the sub gradient for lasso
def compute_sub_grad(x_k,A,b,tau):
    grad_l1 = tau*np.sign(x_k)
    grad_l2 = A.T.dot(A.dot(x_k)-b)
    grad = grad_l1 + grad_l2
    return grad


def main():
    #initialization
    m = 100
    n = 500
    s = 5
    A = np.random.randn(m,n)
    xs = np.zeros([n,1])
    picks = np.random.permutation(np.arange(1,n))
    xs[picks[0:s],0] = np.random.randn(s,1).flatten()
    b = A.dot(xs)

    x_k = np.zeros([n,1])
    epsilon = 1e-2
    tau = 1

    k = 1
    error = []

    #sub gradient method
    while np.linalg.norm(x_k-xs)/np.linalg.norm(xs) >= epsilon:
        t_k = 0.05/k
        grad = compute_sub_grad(x_k,A,b,tau)
        x_k = x_k - t_k*grad
        k += 1
        print(np.linalg.norm(x_k-xs)/np.linalg.norm(xs))
        error.append(np.linalg.norm(x_k-xs)/np.linalg.norm(xs))

    '''
    plt.figure()
    plt.plot(error)
    plt.show()
    '''

if __name__ == "__main__":
    main()