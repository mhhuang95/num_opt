import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def mu(t):
    tau0 = 330
    return min(1-np.exp(-t/tau0), 0.2)

def main():
    n = 128
    x = (np.random.randn(n,1) + 1j* np.random.randn(n,1)) * (np.random.rand(n,1) < 0.04)

    m = int(4*n)
    A = 1 / np.sqrt(2) * np.random.randn(m, n) + 1j / np.sqrt(2) * np.random.randn(m, n)

    y = np.abs(A.dot(x))** 2

    npower_iter = 50
    z0 = np.random.randn(n,1)
    z0 = z0 / np.linalg.norm(z0)
    for tt in range(npower_iter):
        z0 = np.conj(A.T).dot(y* (A.dot(z0)))
        z0 = z0/np.linalg.norm(z0)

    normest = np.sqrt(np.sum(y) / y.shape[0])
    z = normest * z0

    Relerrs = [np.linalg.norm(x - np.exp(-1j * np.angle(np.trace(np.conj(x.T).dot(z)))) * z)/np.linalg.norm(x)]
    print(Relerrs[-1])


    T = 2500

    for t in range(T):
        yz =  A.dot(z)
        grad = 1/m*np.conj(A.T).dot((np.abs(yz)**2-y)* yz)
        z = z - mu(t) / normest**2*grad
        Relerrs.append(np.linalg.norm(x - np.exp(-1j * np.angle(np.trace(np.conj(x.T).dot(z)))) * z) / np.linalg.norm(x))
        print(t,Relerrs[-1])

    plt.figure()
    plt.semilogy(Relerrs)
    plt.show()

if __name__ == "__main__":
    main()

