import Tools
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[-1, 0],
                 [0, 1],
                 [2, 9],
                 [3, 25],
                 [4, 67]])


f = Tools.lagrange(data)
x = np.arange(-2, 5, 0.05)
y = [f(i) for i in x]

plt.plot(x, y, lw=1)


plt.plot(data[:, 0], data[:, 1], "r.")
plt.show()


