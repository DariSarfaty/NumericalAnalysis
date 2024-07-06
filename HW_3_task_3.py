import Tools
import numpy as np

data1 = np.array([[0, 0],
                 [1, 1],
                 [2, 4],
                 [3, 1],
                 [4, 0],
                 [5, 1]])

res = 0.005


Tools.cubic_spline(data1, res)

data2 = np.array([[3, 4],
                 [2, 3],
                 [2.5, 1],
                 [4, 2],
                 [5, 3.5],
                 [4, 4.5]])

Tools.cubic_spline(data2, res)

