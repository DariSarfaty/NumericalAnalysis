import numpy as np
import matplotlib.pyplot as plt


def cubic_spline(x, y):
    # checking for compatibility and setting n
    m = len(x)
    if m != len(y):
        return -1
    n = m - 1

    # initializing arrays
    h = np.zeros(n)
    alpha = np.zeros(n - 1)
    l = np.ones(m)
    mu = np.zeros(n)
    z = np.zeros(m)
    a = y.copy()
    b = np.zeros(m)
    c = np.zeros(m)
    d = np.zeros(m)

    # calculating h
    for i in range(n):
        h[i] = x[i + 1] - x[i]

    # calculating alpha
    for i in range(1, n):
        alpha[i - 1] = (3 / h[i]) * (a[i + 1] - a[i]) - (3 / h[i - 1]) * (a[i] - a[i - 1])

    # tridiagonal system setup
    l[0] = 1
    mu[0] = 0
    z[0] = 0

    for i in range(1, n):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i]

    l[n] = 1
    z[n] = 0
    c[n] = 0

    # solving the system
    for i in range(n-1, -1, -1):
        c[i] = z[i] - mu[i] * c[i + 1]
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    # returning
    return a, b, c, d

if __name__ == "__main__":

    res = 0.1
    x = [0, 1000, 2600, 4600, 6000]
    z = [2600, 4000, 3200, 3600, 2400]
    a, b, c, d = cubic_spline(x, z)
    for i in range(len(x)-1):
        def f(p):
            return a[i] + b[i]*(p - x[i]) + c[i]*(p - x[i])**2 + d[i]*(p - x[i])**3


        xs = np.arange(x[i], x[i + 1] + res, res)
        zs = [f(point) for point in xs]
        plt.plot(xs, zs, "b")

    plt.plot(x,z, "r.")
    plt.show()
