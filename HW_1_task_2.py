"""
written by dari sarfaty :)
"""


import Tools
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator


def f(coor):
    x = coor[0]
    y = coor[1]
    return 4 * y * y + 4 * y - 52 * x - 19


def g(coor):
    x = coor[0]
    y = coor[1]
    return 169 * x * x + 3 * y * y - 111 * x - 10 * y


print(f"found the following solution: {Tools.newton(f, g, [-0.01, -0.01], 0.00000001)}")
"""for some reason epsilon isn't factored in well"""



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-10, 10, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sin(4*Y)*np.cos(0.5*X)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.5, 1.5)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()