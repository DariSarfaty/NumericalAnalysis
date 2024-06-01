def synthetic_division(polynomial, x1, epsilon):
    """finds the root of a given polynomial 'nearest' to the initial guess x1"""
    convergence = 2 * epsilon
    while convergence > epsilon:
        x0 = x1
        c = division(polynomial, x0)
        d = division(c, x0)
        r0 = polynomial[0] + (x0 * c[0])
        r1 = c[0] + (x0 * d[0])
        x1 = x0 - r0/r1
        convergence = abs(x0 - x1)
    return x1


def division(polynomial, x0):
    """divides the polynomial by (x-x0)"""
    a = [0]
    for i in range(len(polynomial) - 1):
        a.append(polynomial[-(i + 1)] + a[i] * x0)
    del a[0]
    a.reverse()
    return a

def all_roots(polynomial, x1, epsilon):
    """finds all roots of a polynomial using synthetic division and reduction of the order,
     the final root (of a linear equation) is found analytically"""
    roots = []
    while len(polynomial) > 2:
        root = synthetic_division(polynomial, x1, epsilon)
        roots.append(root)
        polynomial = division(polynomial, root)
    roots.append(-polynomial[0]/polynomial[1])
    return roots
