import Tools
import numpy as np
import matplotlib.pyplot as plt
res = 0.05
data = np.array([[-1, 0],
                 [0, 1],
                 [2, 9],
                 [3, 25],
                 [4, 67]])


f = Tools.lagrange(data)
x = np.arange(data[0, 0], data[-1, 0] + res, res)
y = [f(i) for i in x]

plt.figure(1)
plt.plot(x, y, lw=1)
plt.plot(data[:, 0], data[:, 1], "r.")
plt.show()

plt.figure(2)
data1 = data[:3]
data2 = data[2:]
g = Tools.lagrange(data1)
h = Tools.lagrange(data2)
x = np.arange(data1[0, 0], data1[-1, 0] + res, res)
y = y = [g(i) for i in x]
plt.plot(x, y, lw=1)
x = np.arange(data2[0, 0], data2[-1, 0] + res, res)
y = y = [h(i) for i in x]
plt.plot(x, y, "g", lw=1)
plt.plot(data[:, 0], data[:, 1], "r.")
plt.show()
