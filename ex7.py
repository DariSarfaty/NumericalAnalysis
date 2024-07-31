import numpy as np
import Tools
import matplotlib.pyplot as plt


def S(x, y):
    return 10 - 2*x


def M(x):
    return 10*x - x**2


plt.figure(1)


step = 0.5
coor = np.arange(0, 10 + step, step)
val = [Tools.RK2(S, 0, 0, i, step) for i in coor]
plt.plot(coor, val, "b")


step = 0.25
coor = np.arange(0, 10 + step, step)
val = [Tools.euler(S, 0, 0, i, step) for i in coor]
plt.plot(coor, val, "r")


step = 0.05
coor = np.arange(0, 10 + step, step)
val = [Tools.euler(S, 0, 0, i, step) for i in coor]
plt.plot(coor, val, "g")


val = [M(i) for i in coor]
plt.plot(coor, val, "y")

plt.show()
