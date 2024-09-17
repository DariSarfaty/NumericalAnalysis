import numpy as np
import bisect
import matplotlib.pyplot as plt
import project_tools

# Set resolution and data points
res = 100.0
x = [0, 1000, 2600, 4600, 6000]
z = [2600, 4000, 3200, 3600, 2400]


# Calculate spline coefficients
a, b, c, d = project_tools.cubic_spline(x, z)

# Prepare spline functions
functions = []
for i in range(len(x) - 1):
    def f(p, i=i):
        return a[i] + b[i] * (p - x[i]) + c[i] * (p - x[i]) ** 2 + d[i] * (p - x[i]) ** 3
    functions.append(f)

# Spline function with boundary checking
def spline(p):
    # Find the interval where p belongs
    idx = bisect.bisect_left(x, p)
    # Handle out of range cases
    if idx == 0:
        return functions[0](p)
    elif idx >= len(x):
        return functions[-1](p)
    else:
        return functions[idx - 1](p)

# Generate x values and compute corresponding z values
xs = np.arange(x[0], x[-1] + res, res)
zs = [spline(point) for point in xs]

# Plotting
plt.plot(xs, zs, "b", label="Cubic Spline")
plt.plot(x, z, "r.", label="Data Points")
plt.legend()
plt.show()
