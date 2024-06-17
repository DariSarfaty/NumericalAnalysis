import Tools
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('n')
ax.set_ylabel('m')
ax.set_zlabel('time. \n y = LU decomposition\n b = jacobi\n r = row reduction\n g = gauss-seidel')


"""test LU decomposition:"""
ntests = 25
mtests = 50
elaps = np.zeros((ntests,mtests))
for n in range(2, ntests):
    print(n)
    start = []
    end = []
    A = np.random.randint(1, 10, (n, n))
    for i in range(n):
        A[i, i] *= 10
    for m in range(1, mtests):
        C = np.random.rand(n, m)
        start.append(time.time())
        Tools.LU_decomposition(A, C)
        end.append(time.time())
        elaps[n, m] = sum(en - st for en, st in zip(end, start))



# Make data.
X = np.arange(2, ntests)
Y = np.arange(2, mtests)
X, Y = np.meshgrid(X, Y)
Z = elaps[X, Y]

# Plot the surface.

ax.plot_surface(X, Y, Z, color="y")

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.




"""test jacobi:"""
ntests = 25
mtests = 50
elaps = np.zeros((ntests,mtests))
for n in range(2, ntests):
    print(n)
    start = []
    end = []
    A = np.random.randint(1, 10, (n, n))
    C = np.random.rand(n, m)
    for i in range(n):
        A[i, i] *= 10
    for m in range(2, mtests):
        c = np.random.rand(n, 1)
        start.append(time.time())
        Tools.jacobi(A, c, 0.00000001, 100)
        end.append(time.time())
        elaps[n, m] = sum(en - st for en, st in zip(end, start))




# Make data.
X = np.arange(2, ntests)
Y = np.arange(2, mtests)
X, Y = np.meshgrid(X, Y)
Z = elaps[X, Y]

# Plot the surface.

ax.plot_surface(X, Y, Z, color="b")

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.



"""test gauss-seidel:"""
ntests = 25
mtests = 50
elaps = np.zeros((ntests,mtests))
for n in range(2, ntests):
    print(n)
    start = []
    end = []
    A = np.random.randint(1, 10, (n, n))
    C = np.random.rand(n, m)
    for i in range(n):
        A[i, i] *= 10
    for m in range(2, mtests):
        c = np.random.rand(n, 1)
        start.append(time.time())
        Tools.gauss_seidel(A, c, 0.00000001, 100)
        end.append(time.time())
        elaps[n, m] = sum(en - st for en, st in zip(end, start))



# Make data.
X = np.arange(2, ntests)
Y = np.arange(2, mtests)
X, Y = np.meshgrid(X, Y)
Z = elaps[X, Y]

# Plot the surface.

ax.plot_surface(X, Y, Z, color="g")

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.




"""test row reduction:"""
ntests = 25
mtests = 50
elaps = np.zeros((ntests,mtests))
for n in range(2, ntests):
    print(n)
    start = []
    end = []
    A = np.random.randint(1, 10, (n, n))
    C = np.random.rand(n, m)
    for i in range(n):
        A[i, i] *= 10
    for m in range(2, mtests):
        c = np.random.rand(n, 1)
        start.append(time.time())
        Tools.row_reduction(A, c)
        end.append(time.time())
        elaps[n, m] = sum(en - st for en, st in zip(end, start))



# Make data.
X = np.arange(2, ntests)
Y = np.arange(2, mtests)
X, Y = np.meshgrid(X, Y)
Z = elaps[X, Y]

# Plot the surface.

ax.plot_surface(X, Y, Z, color="r")

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.


plt.show()


