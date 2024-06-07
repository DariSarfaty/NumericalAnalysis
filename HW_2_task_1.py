import Tools
import numpy as np

A = np.array([[3, -3, 2, -4],
                    [-2, -1, 3, -1],
                    [5, -2, -3, 2],
                    [-2, 4, 1, 2]], dtype=float)

c = np.array([[7.9], [-12.5], [18.0], [-8.1]])

A = np.array([[30, 3, 2, -4],
                    [-2, -10, 3, -1],
                    [5, -2, -30, 2],
                    [-2, 4, 1, 20]], dtype=float)


c = np.array([[7.9], [12.5], [18.0], [8.1]])
"""A = np.array([[4, -1, 0, 0],
              [-1, 4, -1, 0],
              [0, -1, 4, -1],
              [0, 0, -1, 3]], dtype=float)
c = np.array([15, 10, 10, 10], dtype=float)"""

print(Tools.row_reduction(A, c))
print(Tools.LU_decomposition(A, c))

print(Tools.jacobi(A, c, 0.00000001, 100))

