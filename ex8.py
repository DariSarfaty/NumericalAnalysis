import math

import numpy as np
import Tools
import matplotlib.pyplot as plt

x0 = 0.7
v0 = 0
k = 40
m = 2
w = math.sqrt(k/m)


def A(x):
    return -1*(k/m)*x


def f(x, y, v):
    return A(y)


def V(t):
    return -1 * x0 * w * math.sin(w*t) + v0 * math.cos(w*t)


def X(t):
    return (v0/w) * math.sin(w*t) + x0 * math.cos(w*t)

plt.figure(1)
plt.title("leapfrog")

step = 0.1
coor = np.arange(0, 2.5 + step, step)
val1, val2 = zip(*[Tools.leapfrog(A, x0, v0, i, step) for i in coor])
plt.plot(coor, val1, "b")
plt.plot(coor, val2, "r")


plt.figure(2)
plt.title("euler 0.1s")
val1, val2 = zip(*[Tools.euler_2nd(f, 0, x0, v0, i, step) for i in coor])
plt.plot(coor, val1, "b")
plt.plot(coor, val2, "r")


plt.figure(5)
plt.title("analytical")
xs = [X(i) for i in coor]
vs = [V(i) for i in coor]
plt.plot(coor, xs, "b")
plt.plot(coor, vs, "r")



plt.show()
