import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Lx = 10        
Ly = 10        
Nx = 128        # Number of grid points
Ny = 128      
alpha = 0.01    # Thermal diffusivity
Tmax = 1000     
dt = 0.01    
dx = Lx / Nx    
dy = Ly / Ny 
Nt = int(Tmax / dt)  # Number of time steps

# Discretize spatial domain
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)

# Initial condition
u0 = np.zeros((Ny, Nx)) 
u0[:, 0] = 1       

# frequency domain
kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
kx, ky = np.meshgrid(kx, ky)
u_new = u0

def get_heat_eq_solution(u_new, kx, ky, Nt, dt):
    ani = []
    for _ in range(Nt):
        # forward Fourier transform
        u_hat = np.fft.fft2(u_new)

        # heat equation in Fourier space
        u_hat *= np.exp(-alpha * (kx**2 + ky**2) * dt)
        
        # put back to spatial space
        u_new = np.fft.ifft2(u_hat)
        
        # impose boundary conditions
        u_new[:, 0] = 1
        u_new[:, -1] = 0
        u_new[0, :] = 0
        u_new[-1, :] = 0
        if _ % 50 == 0:
            ani.append([plt.imshow(u_new.real, cmap='hot', animated=True)])
            print('rendering frame', _)
    return ani
fig = plt.figure()
result = get_heat_eq_solution(u_new, kx, ky, Nt, dt)
anima = animation.ArtistAnimation(fig, result, interval=100, blit=True)
anima.save('./results/heat_eq.gif', dpi=80, writer='pillow')
plt.show()