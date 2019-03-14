
import numpy as np

def compute_grad(x_k, A, b):
    grad_l2 = np.conj(A.T).dot(A.dot(x_k) - b)          #Wirtinger derivative of  ||Ax-b||_2^2
    return grad_l2

def f(x_k,A,b):

    return 0.5*np.linalg.norm(A.dot(x_k)-b)**2


def F(x_k,A,b,tau):
    return 0.5 * np.linalg.norm(A.dot(x_k) - b) ** 2 + tau*np.sum(np.abs(x_k))

def model(x,xk,A,b,GammaK):

    innerProd = compute_grad(xk,A,b).T.dot(x - xk)
    xDiff = x - xk
    return f(xk,A,b) + innerProd + (1.0/(2.0*GammaK))*xDiff.T.dot(xDiff)


def main():
    # initialization
    m = 100                                             #Number of measurements
    n = 500
    s = 5
    A = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    xs = np.zeros([n, 1], dtype=complex)
    picks = np.random.permutation(np.arange(1, n))
    xs[picks[0:s], 0] = np.random.randn(s, 1).flatten() + 1j * np.random.randn(s, 1).flatten()
    b = A.dot(xs)                                       #b = Ax

    x_k = np.random.rand(n, 1) + 1j * np.random.rand(n, 1)
    x_k /= np.linalg.norm(x_k)

    epsilon = 1e-2
    beta = 0.7
    tau = 0.01

    while np.linalg.norm(xs - x_k)/np.linalg.norm(xs) >= epsilon:
        t = 0.1
        while f(x_k - t * compute_grad(x_k, A, b), A, b) > model(x_k - t * compute_grad(x_k, A, b), x_k, A, b, t):
            t = beta * t
        x_k = x_k - t * compute_grad(x_k, A, b)
        mask = np.abs(x_k) > 0
        x_k[mask] = np.maximum(np.abs(x_k[mask]) - tau * t, 0) * (x_k[mask] / np.abs(x_k[mask]))
        print(np.linalg.norm(xs -  x_k)/np.linalg.norm(xs))
    print(np.hstack([xs,  x_k]))


if __name__ == "__main__":
    main()