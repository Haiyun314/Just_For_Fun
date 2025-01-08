import numpy as np
import matplotlib.pyplot as plt

# Parameters
L_x = 10  # Length of the domain in x-direction
L_y = 10  # Length of the domain in y-direction
Nx = 50  # Number of spatial points in the x-direction
Ny = 50  # Number of spatial points in the y-direction
alpha = 0.01  # Thermal diffusivity
T_max = 5  # Total time to simulate
dt = 0.01  # Time step size
dx = L_x / (Nx - 1)  # Space step size in x-direction
dy = L_y / (Ny - 1)  # Space step size in y-direction
Nt = int(T_max / dt)  # Number of time steps
num_terms = 20  # Number of Fourier terms (increase for better accuracy)

# Fourier coefficients (calculated from the boundary condition u(x=0, y, t) = 1)
B_m = np.zeros(num_terms)
for m in range(1, num_terms + 1):
    B_m[m - 1] = 2 / L_y * np.trapz(np.sin(m * np.pi * np.linspace(0, L_y, Ny)) , np.linspace(0, L_y, Ny))

# Solution grid setup
x = np.linspace(0, L_x, Nx)
y = np.linspace(0, L_y, Ny)
X, Y = np.meshgrid(x, y)

# Initialize the solution
u = np.zeros((Nx, Ny))
plt.figure()

# Calculate the solution at each time step
for t in range(1, Nt):
    u_new = np.zeros_like(u)
    for n in range(1, num_terms + 1):
        for m in range(1, num_terms + 1):
            # Fourier mode contribution for temperature evolution
            A_nm = B_m[m - 1] * np.sin(n * np.pi * X / L_x) * np.sin(m * np.pi * Y / L_y) * np.exp(
                -alpha * np.pi**2 * (n**2 / L_x**2 + m**2 / L_y**2) * t * dt
            )
            u_new += A_nm
    u = u_new  # Update the temperature field

    # Plot the temperature distribution at every 100 time steps

    plt.clf()
    plt.contourf(X, Y, u.T, cmap='hot', levels=100)
    plt.colorbar()
    plt.title(f"Temperature Distribution at t = {t*dt:.2f}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.pause(0.05)
plt.show()
