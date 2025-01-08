import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# game of life (rules)
# Any live cell with fewer than two live neighbors dies (underpopulation).
# Any live cell with two or three live neighbors lives on to the next generation (survival).
# Any live cell with more than three live neighbors dies (overpopulation).
# Any dead cell with exactly three live neighbors becomes a live cell (reproduction).

root_path = os.path.join(os.path.abspath(__file__), os.pardir, os.pardir)

grid = 100
domain = np.round(np.random.randint(0, 57, size= (grid, grid))/100)  # initialize active points
print(np.sum(domain))

# Create a meshgrid for the 2D plane (u, v)
u = np.linspace(0, 1, grid)
v = np.linspace(0, 1, grid)
u, v = np.meshgrid(u, v)

# create donut
R = 1  #  radius 
r = 0.3  
x = (R + r * np.cos(2 * np.pi * v)) * np.cos(2 * np.pi * u)
y = (R + r * np.cos(2 * np.pi * v)) * np.sin(2 * np.pi * u)
z = r * np.sin(2 * np.pi * v)


fig, ax = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={'projection': '3d'})
# Labels and titles
ax[0].set_axis_off()
ax[1].set_axis_off()

ax[0].set_title("Play game of life on Donut without Donut")
ax[1].set_title("Play game of life on Donut")

def update():
    # game of life
    global domain
    reset_domain = domain.copy()
    for i in range(grid):
        for j in range(grid):
            neibers = np.sum(domain[i-1:i+2, j-1:j+2]) - domain[i, j]
            if neibers >= 4 or neibers < 2:
                reset_domain[i, j] = 0
            elif neibers == 3:
                reset_domain[i, j] = 1
    
    domain = reset_domain
    return domain.astype(np.int0)


def evol(frame):
    ax[0].cla()
    ax[1].cla()

    #project the 2d game of life to 3d donut
    cur = update()
    mask = cur == 1
    x_p = x * cur
    x_p = x_p[mask]
    y_p = y * cur
    y_p = y_p[mask]
    z_p = z * cur
    z_p = z_p[mask]

    # To address rendering issues caused by overlapping points, we apply a slight shift along the z-axis.
    z_p += np.sign(z_p)*0.01 

    ax[0].scatter(x_p, y_p, z_p, color = 'y', s=1)
    ax[1].plot_surface(x, y, z, cmap='hot_r')
    ax[1].scatter(x_p, y_p, z_p, color = 'y', s=1)

    # Labels and titles
    ax[0].set_axis_off()
    ax[1].set_axis_off()

    ax[0].set_title("Play game of life on Donut without Donut")
    ax[1].set_title("Play game of life on Donut")


ani = FuncAnimation(fig, evol, frames= 40)

if not os.path.exists('results'):
    os.makedirs('results')

ani.save('results/donut.gif', writer='pillow')


plt.show()