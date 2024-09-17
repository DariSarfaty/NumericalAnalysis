import numpy as np
import Tools
data1 = np.array([[0, 2600],
                 [1000, 4000],
                 [2600, 3200],
                 [4600, 3600],
                 [6000, 2400]])

res = 0.1


Tools.cubic_spline(data1, res)