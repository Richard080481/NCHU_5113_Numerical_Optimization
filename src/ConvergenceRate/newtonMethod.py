import numpy as np
import signal
import time
import argparse
import pickle
import sys

def pickleMe(data):
    filePath = "data.newton"
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

def foo(x):
    return 100 * ((x[1] - (x[0] ** 2)) ** 2) + (1 - x[0]) ** 2

def fooGradient(x):
    return np.array([200 * (x[0] ** 2 - x[1]) * 2 * x[0] + 2 * x[0] - 2, 200 * (x[1] - x[0] ** 2)])

def fooHessian(x):
    return np.array([[400 *(3 * (x[0] ** 2) - x[1]) + 2, -400 * x[0]], [-400 * x[0], 200]])

def armijo(x, alpha, p, c, gradX):
    f1 = foo(x)
    f2 = foo(x + alpha * p)
    imp = c * alpha * gradX.T @ p
    # print("x=", x, "f1=", f1, "f2=", f2, "imp=", imp, "alpha=", alpha, "p=", p)
    return f2 <= f1 + imp

def happy(gradSquareX):
    return gradSquareX < 1e-8

def RunArmijo(rho=0.9, c=0.3, alpha_0=1):
    x_0 = [0, 0]
    x = list(x_0)
    k = 1
    fooValues = []
    gradValues = []
    # try:
    #     filePath = "data.pickle"
    #     with open(filePath, 'rb') as f:
    #         loadedData = pickle.load(f)
    #         if rho     == loadedData['rho'] and \
    #            x_0     == loadedData['x_0'] and \
    #            c       == loadedData['c']   and \
    #            alpha_0 == loadedData['alpha_0']:
    #             k = loadedData['k']
    #             x = loadedData['x']
    #             fooValues = loadedData['fooValues']
    #             gradValues = loadedData['gradValues']
    #             print(f"start from k={k}")
    # except:
    #     pass

    def packUp():
        return {"x_0":x_0, "k":k, "x":x, "rho":rho, "alpha_0":alpha_0, "c":c, "fooValues":fooValues, "gradValues":gradValues}

    signal.signal(signal.SIGINT, lambda sig, frame : signal_handler(sig, packUp()))

    t1 = time.time_ns()

    while True:
        fooValue = foo(x)
        if k % 1000 == 0:
            t2 = time.time_ns()
            print(f"\rk={k}, foo={fooValue}, time={(t2-t1)/1e9}, x={x}", end="")
        alpha = alpha_0
        fooValues.append(fooValue)
        gradX = fooGradient(x)
        gradSquareX = gradX.T @ gradX
        gradValues.append(gradSquareX)
        if happy(gradSquareX):
            break

        hussInverse = np.linalg.inv(fooHessian(x))
        p = - hussInverse @ gradX

        while not armijo(x, alpha, p, c, gradX):
            alpha = alpha * rho
        # next step
        x = x + alpha * p
        k = k + 1
        # print(gradX)
    print()
    print("k=", k)
    pickleMe(packUp())
    return fooValues

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Armijo algorithm with specified parameters.')
    parser.add_argument('--rho', type=float, default=0.5, help='The value of rho for Armijo rule')
    parser.add_argument('--c', type=float, default=0.5, help='The value of c for Armijo rule')
    parser.add_argument('--alpha_0', type=float, default=1, help='The initial value of alpha')
    args = parser.parse_args()
    RunArmijo(args.rho, args.c, args.alpha_0)
