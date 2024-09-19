import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation
import project_tools
import bisect

res = 1.0
X = [0, 10, 26, 46, 60]
Z = [26, 40, 32, 36, 24]


# Calculate spline coefficients
a, b, c, d = project_tools.cubic_spline(X, Z)

# Prepare spline functions
functions = []
for i in range(len(X) - 1):
    def f(p, i=i):
        return a[i] + b[i] * (p - X[i]) + c[i] * (p - X[i]) ** 2 + d[i] * (p - X[i]) ** 3
    functions.append(f)

# Spline function
def spline(p):
    # Find the interval where p belongs
    idx = bisect.bisect_left(X, p)
    # Handle out of range cases
    if idx == 0:
        return functions[0](p)
    elif idx >= len(X):
        return functions[-1](p)
    else:
        return functions[idx - 1](p)

# Generate x values and compute corresponding z values
xs = np.arange(X[0], X[-1] + res, res)
zs = [spline(point) for point in xs]

# Grid parameters
nx, nz = 60, 60  # Grid size
dx, dz = 1, 1    # Spatial step sizes
dt = 0.01        # Time step size
T = 1.0         # Total simulation time
nt = int(T / dt) # Number of time steps

# Velocity field
v = np.zeros((nx, nz))
for x in range(nx):
    for z in range(nz):
        if z > spline(x):
            v[z, x] = 30
        else:
            v[z, x] = 20

# Source parameters
x_source, z_source = 30, 28  # Source located at (30, 28)

# Source function
def source_function(t):
    if t <= 0.05:
        return t * math.exp(2 * math.pi * t) * math.sin(2 * math.pi * t)
    return 0

# Initialize the wave field (u and u_t)
u = np.zeros((nx, nz))       # Current wave field
u_prev = np.zeros((nx, nz))  # Previous wave field
u_next = np.zeros((nx, nz))  # Next wave field

# Set up the figure and axis for real-time plotting
fig, ax = plt.subplots()

# Plot setup
cax = ax.imshow(u, extent=[0, nx, 0, nz], cmap=plt.cm.bone, origin='lower', animated=True)
fig.colorbar(cax)

# Plot the spline and source location
function_plot, = ax.plot(xs, zs, 'g--')  # Plot the boundary in green
source_plot, = ax.plot(x_source, z_source, 'r.', markersize=10, label='Source')  # Plot the source as a red circle

ax.invert_yaxis()

# Update function for animation
def update(frame):
    global u, u_prev, u_next

    # Apply source at (x_source, z_source)
    t = frame * dt
    u[z_source, x_source] += source_function(t)

    # Finite difference update for the wave equation
    u_next[2:-2, 2:-2] = (2 * u[2:-2, 2:-2] - u_prev[2:-2, 2:-2] + (v[2:-2, 2:-2] ** 2) * dt ** 2 *
                                    ((-u[4:, 2:-2] + 16 * u[3:-1, 2:-2] - 30 * u[2:-2, 2:-2] + 16 * u[1:-3, 2:-2]
                                    - u[:-4, 2:-2]) / (12 * dx ** 2)
                                    + (-u[2:-2, 4:] + 16 * u[2:-2, 3:-1] - 30 * u[2:-2, 2:-2] + 16 * u[2:-2, 1:-3]
                                    - u[2:-2, :-4]) / (12 * dz ** 2)))


    # Update the previous wave field
    u_prev = np.copy(u)
    u = np.copy(u_next)

    # Update the wave field plot
    cax.set_array(u)

    return cax, function_plot, source_plot

# Animation function, where 'interval' is the delay between frames in milliseconds
ani = FuncAnimation(fig, update, frames=range(nt), blit=True, interval=50, repeat=False)

# Show the animation
plt.legend()
plt.show()