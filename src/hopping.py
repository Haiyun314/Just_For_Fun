import matplotlib.pyplot as plt
import numpy as np
import os

# Parameters for the Hopalong Attractor
a, b, c = 1.0, -1.8, 1.0
num_points = 200000

# Initialize arrays to hold the points
x = np.zeros(num_points)
y = np.zeros(num_points)

# Generate the points
for i in range(num_points - 1):
    x[i + 1] = y[i] - np.sign(x[i]) * np.sqrt(abs(b * x[i] - c))
    y[i + 1] = a - x[i]

# Plot the attractor
plt.figure(figsize=(5, 5))
plt.scatter(x, y,cmap= 'hot', s=0.1, c=range(num_points))
plt.axis("off")
plt.title("Hopalong Attractor")
if not os.path.exists('results'):
    os.makedirs('results')
plt.savefig('results/hopalong_attractor.png')
plt.show()
