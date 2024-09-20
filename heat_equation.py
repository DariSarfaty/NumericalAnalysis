import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
Lx, Ly = 1.5, 1.5  # Length of the grid in x and y direction (meters)
dx, dy = 0.05, 0.05  # Spatial step size (meters)
dt = 0.1  # Time step size (seconds)
k = 1.786e-3  # Thermal diffusivity (m^2/s)
sigmax, sigmay = 0.00625, 0.00625  # Standard Deviations
Nx, Ny = int(Lx / dx) + 1, int(Ly / dy) + 1  # Number of grid points in x and y directions


# Time parameters
t_max = 60  # Maximum time (seconds)
Nt = int(t_max / dt)  # Number of time steps

# Initialize the temperature field
T = np.full((Ny, Nx), 10.0)  # Initial temperature T(x, y, 0) = 10

# Boundary conditions
def apply_boundary_conditions(T):

    for i in range(Ny):
        T[i, 0] = 100 - 112.5 * i * dy
        T[i, -1] = 100 - 60 * i * dy
    T[int(0.8 / dy) + 1:, 0] = 10
    T[0, :] = 100
    T[-1, :] = 10

apply_boundary_conditions(T)

# Heat source function
def heat_source(x, y, t):
    return -1e4 * np.exp(-(((x - 1)**2) / (2 * sigmax**2) + ((y - 0.5)**2) / (2 * sigmay**2)) - 0.1 * t)

# Animation function
def update_temperature(T, T_new, t):
    for i in range(1, Ny-1):
        x = i * dx
        for j in range(1, Nx-1):
            y = j * dy
            d2T_dx2 = (T[i, j+1] - 2 * T[i, j] + T[i, j-1]) / dx**2
            d2T_dy2 = (T[i+1, j] - 2 * T[i, j] + T[i-1, j]) / dy**2
            F = heat_source(y, x, t)
            T_new[i, j] = T[i, j] + dt * (k * (d2T_dx2 + d2T_dy2) + F)
    apply_boundary_conditions(T_new)
    return T_new

# Create the figure and axis for the animation
fig, ax = plt.subplots()
cax = ax.imshow(T, cmap='hot', origin='lower', extent=[0, Lx, 0, Ly], vmin=0, vmax=100)

# Add colorbar and set the label
colorbar = fig.colorbar(cax)
colorbar.set_label('Temperature (°C)')  # Add your label here

ax.set_title('Temperature Distribution')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')



# Update function for FuncAnimation
def animate(n):
    global T
    T_new = T.copy()
    T = update_temperature(T, T_new, n * dt)
    cax.set_data(T)
    ax.set_title(f'Temperature at t = {n * dt:.1f} s')
    return cax,

# Create the animation
anim = FuncAnimation(fig, animate, frames=Nt, interval=50, blit=False, repeat=False)

plt.show()

# Save the animation
anim.save('heat_equation_solution.gif', writer='ffmpeg', fps=10)  # Save as MP4 with FFmpeg