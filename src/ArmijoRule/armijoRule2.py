import numpy as np
import signal
import sys
import time
import argparse
import pickle

def pickleMe(data):
    filePath = "data.pickle"
    with open(filePath, 'wb') as f:
        pickle.dump(data, f)

def signal_handler(sig, f_locals):
    k       = f_locals['k']
    x       = f_locals['x']
    # rho     = f_locals['rho']
    # m       = f_locals['m']
    # alpha_0 = f_locals['alpha_0']
    # c       = f_locals['c']
    print(f"\nk={k}\n")

    pickleMe(f_locals)
    sys.exit(0)

def foo(x, A, b):
    temp = A @ x - b
    return temp.T @ temp

def fooGradient(x, A, b):
    return 2 * (A.T @ ((A @ x) - b))

def armijo(x, alpha, p, c, gradSquareX, A, b):
    f1 = foo(x, A, b)
    f2 = foo(x + alpha * p, A, b)
    imp = c * alpha * -gradSquareX
    return f2 <= f1 + imp

def happy(gradSquareX):
    return gradSquareX < 1e-8

def RunArmijo(m, rho=0.9, c=0.5, alpha_0=1):
    np.random.seed(4111064232)
    A = np.random.rand(m, m)
    b = np.random.rand(m, 1)
    x_0 = np.random.rand(m, 1)
    x = np.array(x_0)
    k = 1
    fooValues = []
    filePath = "data.pickle"
    try:
        with open(filePath, 'rb') as f:
            loadedData = pickle.load(f)
            if m       == loadedData['m']   and \
               rho     == loadedData['rho'] and \
               c       == loadedData['c']   and \
               alpha_0 == loadedData['alpha_0']:
                k = loadedData['k']
                x = loadedData['x']
                fooValues = loadedData['fooValues']
                print(f"start from k={k}")
    except:
        pass

    def packUp():
        return {"k":k, "x":x, "rho":rho, "m":m, "alpha_0":alpha_0, "c":c, "fooValues":fooValues}

    signal.signal(signal.SIGINT, lambda sig, frame : signal_handler(sig, packUp()))
    t1 = time.time_ns()

    while True:
        fooValue = foo(x, A, b).item()
        if k % 1000 == 0:
            t2 = time.time_ns()
            print(f"\rk={k}, foo={fooValue}, time={(t2-t1)/1e9}", end="")
        alpha = alpha_0
        fooValues.append(fooValue)
        gradX = fooGradient(x, A, b)
        gradSquareX = gradX.T @ gradX
        if happy(gradSquareX):
            break
        p = -gradX
        while not armijo(x, alpha, p, c, gradSquareX, A, b):
            alpha = alpha * rho
        x = x + alpha * p
        k = k + 1
    print()
    print(x)
    pickleMe(packUp())
    return fooValues

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Armijo algorithm with specified parameters.')
    parser.add_argument('--m', type=int, default=32, help='The shape of the matrix A')
    parser.add_argument('--rho', type=float, default=0.5, help='The value of rho for Armijo rule')
    parser.add_argument('--c', type=float, default=0.5, help='The value of c for Armijo rule')
    parser.add_argument('--alpha_0', type=float, default=1, help='The initial value of alpha')
    args = parser.parse_args()
    print(RunArmijo(args.m, args.rho, args.c, args.alpha_0))
