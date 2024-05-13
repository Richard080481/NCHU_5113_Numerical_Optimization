import matplotlib.pyplot as plt
from armijoRule2 import RunArmijo

y = RunArmijo(m=20, rho=0.5, c=0.7, alpha_0=1)
# xStart = int(0 if len(y) < 1e5 else 1e3)
xStart = 0
for i in range(len(y)):
    if y[i] < 0.01:
        xStart = i
        break
x = [i for i in range(len(y))]

# Create a plot
plt.plot(x[xStart:], y[xStart:], marker='o', linestyle='-')

# Add title and labels
plt.title('Armijo Rule')
plt.xlabel('iteration')
plt.ylabel('$||Ax-b||$')

# Show the plot
plt.grid(True)  # Show grid
plt.show()
