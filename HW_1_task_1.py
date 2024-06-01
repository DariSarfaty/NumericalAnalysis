import Bisection
import NewtonRaphsonSecant
import SyntheticDivision
import PolynomialToFunction

polynomial = [3,0,-7,2,1]
interval = [-5, 3]
epsilon = 0.0001
f = PolynomialToFunction.polynomial_to_function(polynomial)

"""find initial guess using bisection:"""
initial_root = Bisection.bisection(f, interval, 0.1)
"""find a better approximation using the other methods:"""
root1 = NewtonRaphsonSecant.newton_raphson_secant(f, initial_root, epsilon)
root2 = SyntheticDivision.synthetic_division(polynomial, initial_root, epsilon)

print(initial_root, root1, root2)


