import Tools
import numpy as np


A = np.array([[3, -3, 2, -4],
                    [-2, -1, 3, -1],
                    [5, -2, -3, 2],
                    [-2, 4, 1, 2]], dtype=float)
n, m = np.shape(A)

c = np.array([[7.9], [-12.5], [18.0], [-8.1]])

functions = {}
functions["Gauss-Seidel"] = Tools.gauss_seidel
functions["LU Decomposition"] = Tools.LU_decomposition
functions["Row Reduction"] = Tools.row_reduction

func = Tools.selectFromDict(functions, "Method")

print(func(A, c))


