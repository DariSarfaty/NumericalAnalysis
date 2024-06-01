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

evaluation = True in [abs(r - root) <= epsilon for r in true_roots]

print(f"The true roots of the function are:")
for r in true_roots:
    print(r)
print(f"The calculated root is {root} \nIs it within the allowed error? {evaluation}")


root2 = SyntheticDivision.synthetic_division(polynomial, initial_root, epsilon)
