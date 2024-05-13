import numpy as np
import signal
import sys
import time
import time

def signal_handler(sig, frame):
    global x, k
    print(f"\nk={k}, x={x}")
    sys.exit(0)

def foo(x):
    global A, b, fooCounter
    temp = A @ x - b
    fooCounter += 1
    return temp.T @ temp

def fooGradient(x):
    global A, b, fooGrCounter
    fooGrCounter += 1
    return 2 * (A.T @ ((A @ x) - b))

def armijo(x, alpha, p, c, gradSquareX):
    global armijoCounter
    armijoCounter += 1
    f1 = foo(x)
    f2 = foo(x + alpha * p)
    imp = c * alpha * -gradSquareX
    # print("f1=", f1, "f2=", f2, "imp=", imp, "x=", x, "alpha=", alpha, "p=", p)
    return f2 <= f1 + imp

def happy(gradSquareX):
    global k
    return gradSquareX < 1e-8

fooCounter = 0
fooGrCounter = 0
armijo43Counter = 0

m = 32 if len(sys.argv) == 1 else int(sys.argv[1]) # the shape of the matrix A
np.random.seed(4111064232)
A = np.random.rand(m, m) # an m-by-m matrix
b = np.random.rand(m, 1)
x_0 = np.random.rand(m, 1)
x = np.array(x_0)
k = 1

# A = np.array([[1, 2], [3, 4]])
# x_sol = np.array([[3], [2]])
# b = A @ x_sol
# x = np.array([[0], [0]])
def RunArmijo(rho = 0.9, c = 0.5, alpha_0 = 1):
    global A, b, x_0, fooCounter, fooGrCounter, armijoCounter, k, x
    signal.signal(signal.SIGINT, signal_handler)
    fooValues = []
    fooCounter = 0
    fooGrCounter = 0
    armijoCounter = 0
    x = np.array(x_0)
    t1 = time.time_ns()
    while True:
        fooValue = foo(x).item()
        if k % 1000 == 0:
            t2 = time.time_ns()
            print(f"\rk={k}", f"foo={fooValue}", f"dt={t2-t1}", end = "")
            # t1 = t2
        alpha = alpha_0
        # print(k, foo(x), happy(x), np.linalg.norm(fooGradient(x), 2), alpha)
        fooValues.append(fooValue)
        gradX = fooGradient(x)
        gradSquareX = gradX.T @ gradX
        if happy(gradSquareX):
            break

        # Determin the steepest direction
        p = -gradX

        # Determin alpha with armijoRule
        while not armijo(x, alpha, p, c, gradSquareX):
            alpha = alpha * rho
            # assert alpha > 1e-12

        #next step
        x = x + alpha * p
        k = k + 1

    # print(x, foo(x), k, fooCounter, fooGrCounter, armijoCounter)
    print()
    print(x)
    return fooValues

if __name__ == "__main__":
    print(RunArmijo(rho = 0.5, c = 0.5, alpha_0 = 1))