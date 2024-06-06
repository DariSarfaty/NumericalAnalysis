import Tools
import numpy as np

A = np.array([[3, -3, 2, -4],
                    [-2, -1, 3, -1],
                    [5, -2, -3, 2],
                    [-2, 4, 1, 2]], dtype=float)

c = np.array([[7.9], [-12.5], [18.0], [-8.1]])


B = np.array([[1, 3, 2],
                    [2, 4, 3],
                    [3, 4, 7],], dtype=float)


print(Tools.row_reduction(A, c))
print(Tools.LU_decomposition(A, c))

