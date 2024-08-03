import math
import numpy as np
import Tools
import matplotlib.pyplot as plt

"""I put part b first because it only has 1 figure so when you close it you see all 6 figures of part a :)"""


def g(x, y):
    return -2*y/(1+x)


step = 0.5
x_end = 20
coor = np.arange(0, x_end + step, step)
plt.figure(7)
plt.title("adams part b")
adams_y = Tools.adams_first(g, 0, 2, x_end, step)
plt.plot(coor, adams_y, "b")
plt.show()


x0 = 0.7
t_end = 2.5
v0 = 0
k = 40
m = 2
w = math.sqrt(k/m)


def A(x):
    return -1*(k/m)*x


def f(x, y, v):
    return A(y)

def g(x, y):
    return A(y)

def V(t):
    return -1 * x0 * w * math.sin(w*t) + v0 * math.cos(w*t)


def X(t):
    return (v0/w) * math.sin(w*t) + x0 * math.cos(w*t)


step = 0.1
coor = np.arange(0, t_end + step, step)

plt.figure(1)
plt.title("leapfrog")
val1, val2 = zip(*[Tools.leapfrog(A, x0, v0, i, step) for i in coor])
plt.plot(coor, val1, "b")
plt.plot(coor, val2, "r")


plt.figure(2)
plt.title("euler 0.1s")
val1, val2 = zip(*[Tools.euler_second(f, 0, x0, v0, i, step) for i in coor])
plt.plot(coor, val1, "b")
plt.plot(coor, val2, "r")

plt.figure(3)
plt.title("RK4 0.1s")
val1, val2 = zip(*[Tools.RK4_second(f, 0, x0, v0, i, step) for i in coor])
plt.plot(coor, val1, "b")
plt.plot(coor, val2, "r")

step = 0.05
coor = np.arange(0, t_end + step, step)


plt.figure(4)
plt.title("euler 0.05s")
val1, val2 = zip(*[Tools.euler_second(f, 0, x0, v0, i, step) for i in coor])
plt.plot(coor, val1, "b")
plt.plot(coor, val2, "r")

plt.figure(5)
plt.title("analytical")
xs = [X(i) for i in coor]
vs = [V(i) for i in coor]
plt.plot(coor, xs, "b")
plt.plot(coor, vs, "r")

plt.figure(6)
plt.title("adams")
adams_y, adams_v = Tools.adams(g, f, 0, x0, v0, t_end, step)
plt.plot(coor, adams_y, "b")
plt.plot(coor, adams_v, "r")

plt.show()



