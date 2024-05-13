import matplotlib.pyplot as plt
from steepestGradient import RunArmijo
import pickle
import math
import numpy as np

# xStart = int(0 if len(y) < 1e5 else 1e3)
xStart = 0
y = None
try:
    filePath = "data.pickle"
    with open(filePath, 'rb') as f:
        loadedData = pickle.load(f)
        gradValues = loadedData['gradValues']
        y = list(map(math.log10, gradValues))
        # y = fooValues
    print("Open the data.pickle")
except:
    y = RunArmijo(rho = 0.7, c = 0.5)
    print("Call RunArmijo")

x = [i for i in range(len(y))]

A = np.array([[802, -400],
                [-400, 200]])
# Compute the eigenvalues
eigenvalues = np.linalg.eigvals(A)
print("Eigenvalues:", eigenvalues)
r = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + eigenvalues[1])
print(r)
y_2 =  [i * math.log10(r) for i in range(len(y))]
# Create a plot
plt.plot(x[xStart:], y[xStart:], marker='o', linestyle='-', label='gradient')
plt.plot(x[xStart:], y_2, marker='o', linestyle='-', label='$log(r^k)$')

# Add title and labels
plt.title('Steepest Gradient')
plt.xlabel('iteration')
plt.ylabel('f(x)')
plt.legend()

# Show the plot
plt.grid(True)  # Show grid
plt.show()