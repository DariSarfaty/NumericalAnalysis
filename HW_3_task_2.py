import Tools
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return np.cos(x*4*np.pi)


interval = [0, 0.5]
n = 4 # by changing the value of n to 8 we get very close to the real function as shown in 'task 2 b'
res = 0.005


dis = (interval[1] - interval[0]) / (n-1)
xs = np.arange(interval[0], interval[1]+dis, dis)
ys = np.array([func(x) for x in xs])
data = np.column_stack((xs, ys))
interpolation = Tools.lagrange(data)

myx = input("please enter an x value ")
print("selected value: ", myx)
myx = float(myx)
myy = interpolation(myx)
realy = func(myx)
print(f"the calculated value is {myy}, the real value is {realy}")


x = np.arange(data[0, 0], data[-1, 0] + res, res)
y = [interpolation(i) for i in x]
plt.plot(x, y, lw=1)
x = np.arange(data[0, 0], data[-1, 0] + res, res)
y = [func(i) for i in x]
plt.plot(x, y, "g", lw=1)
plt.plot(xs, ys, "r.")
plt.show()
