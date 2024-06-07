import Tools
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

"""A = np.array([[3, -3, 2, -4],
                    [-2, -1, 3, -1],
                    [5, -2, -3, 2],
                    [-2, 4, 1, 2]], dtype=float)

c = np.array([[7.9], [-12.5], [18.0], [-8.1]])"""

"""A = np.array([[30, 3, 2, -4],
                    [-2, -10, 3, -1],
                    [5, -2, -30, 2],
                    [-2, 4, 1, 20]], dtype=float)


c = np.array([[7.9], [-12.5], [18.0], [-8.1]])"""

ns = []
ms = []
elaps = []
end = []
start = []
for test in range(100):
    n = np.random.randint(2, 10)
    m = np.random.randint(2, 10)
    A = np.random.randint(1, 10, (n, n))
    for i in range(n):
        A[i, i] *= 10
    start.append(time.time())
    for rnd in range(m):
        c = np.random.rand(n, 1)
        Tools.row_reduction(A, c)
    end.append(time.time())
    elaps.append(end[test] - start[test])
    ms.append(m)
    ns.append(n)

ns = np.array(ns)
ms = np.array(ms)
elaps = np.array(elaps)

ms, ns = np.meshgrid(ms, ns)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(ms, ns, elaps, c='y', marker='o')

ax.set_xlabel('m value')
ax.set_ylabel('n value')
ax.set_zlabel('elapsed time')

plt.show()







"""print(Tools.row_reduction(A, c))
print(Tools.LU_decomposition(A, c))
print(Tools.jacobi(A, c, 0.00000001, 100))
print(Tools.gauss_seidel(A, c, 0.00000001, 100))
"""
