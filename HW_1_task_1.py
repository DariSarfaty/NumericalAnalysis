"""
written by dari sarfaty :)
"""


import Tools
import matplotlib.pyplot as plt
import numpy as np


polynomial = [3, 0, -7, 2, 1]
interval = [-5, 3]
true_roots = [-3.7912878475, -0.6180339887, 0.7912878475, 1.6180339887]
epsilon = 0.0001
f = Tools.polynomial_to_function(polynomial)

"""find initial guess using bisection:"""
initial_root = Tools.bisection(f, interval, 0.1)
"""find a better approximation using Newton-Raphson:"""
root = Tools.newton_raphson_secant(f, initial_root, epsilon)
"""is the root withing error range?"""
evaluation1 = True in [abs(r - root) <= epsilon for r in true_roots]

"""find all roots using synthetic division and reduction"""
roots = Tools.all_roots(polynomial, initial_root, epsilon)
roots.sort()
"""are they within error range?"""
evaluation2 = all([abs(true - root) <= epsilon for true, root in zip(true_roots, roots)])

if all([f(x) < epsilon for x in true_roots]):
    print(f"The true roots of the function are:")
    for r in true_roots:
        print(r)
    print(f"The root calculated using bisection and newton-raphson is {root} \nIs it within the allowed error? {evaluation1}")

    print("The roots calculated using synthetic division are:")
    for r in roots:
        print(r)
    print(f"are they within the allowed error? {evaluation2}")
else:
    print("You chose a different function.")
    print(f"The root calculated using bisection and newton-raphson is {root}")
    print("The roots calculated using synthetic division are:")
    for r in roots:
        print(r)


t = np.arange(-5.0, 3.0, 0.01)
s = f(t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='x', ylabel='y',
       title='f(x)')
ax.grid()


plt.show()