"""
written by dari sarfaty :)
"""


import Bisection
import NewtonRaphsonSecant
import SyntheticDivision
import PolynomialToFunction

polynomial = [3, 0, -7, 2, 1]
interval = [-5, 3]
true_roots = [-3.7912878475, -0.6180339887, 0.7912878475, 1.6180339887]
epsilon = 0.0001
f = PolynomialToFunction.polynomial_to_function(polynomial)

"""find initial guess using bisection:"""
initial_root = Bisection.bisection(f, interval, 0.1)
"""find a better approximation using Newton-Raphson:"""
root = NewtonRaphsonSecant.newton_raphson_secant(f, initial_root, epsilon)
"""is the root withing error range?"""
evaluation1 = True in [abs(r - root) <= epsilon for r in true_roots]

print(f"The true roots of the function are:")
for r in true_roots:
    print(r)
print(f"The root calculated using bisection and newton-raphson is {root} \nIs it within the allowed error? {evaluation1}")

"""find all roots using synthetic division and reduction"""
roots = SyntheticDivision.all_roots(polynomial, initial_root, epsilon)
roots.sort()
"""are they within error range?"""
evaluation2 = all([abs(true - root) <= epsilon for true, root in zip(true_roots, roots)])

print("The roots calculated using synthetic division are:")
for root in roots:
    print(root)
print(f"are they within the allowed error? {evaluation2}")
