import numpy as np


def bisection(function, interval, epsilon):
    """finds a root of a function under these conditions:
        0. a root that is also an extrema does not count and cannot be found
        1. given an interval with only one root, it will be returned
        2. given an interval with an odd number of roots, one will be returned
        3. at any other situation the function may not be able to find a root"""
    a = interval[0]
    b = interval[1]
    u = function(a)
    v = function(b)
    if u == 0:
        return a
    elif v == 0:
        return b
    else:
        while abs(a-b) > epsilon:
            c = (a+b)/2
            w = function(c)
            if w == 0:
                return c
            elif u*w < 0:
                b = c
                v = w
            elif v*w < 0:
                a = c
                u = w
            else:
                return "your function may not have exactly one root"
        return c


def newton_raphson_secant(f, x0, epsilon):
    """finds the roots of a function using the newton raphson method"""
    cur = x0 + epsilon
    last = x0
    while abs(cur - last) >= epsilon:
        tag = secant(f, last, cur)
        last = cur
        cur = cur - f(cur)/tag
    return cur


def secant(f, x0, x1):
    """approximates the derivative of f"""
    return (f(x0) - f(x1)) / (x0 - x1)


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


def polynomial_to_function(polynomial):
    """
    disclaimer: I used chatGPT for this function only as I was not sure this was possible in python
    Generate a polynomial function based on the given coefficients.

    Arguments:
    polynomial (list): List of coefficients of the polynomial function,
                         where coefficients[i] corresponds to the coefficient of x^i.

    Returns:
    function: A polynomial function that takes a single argument x.
    """

    def function(x):
        result = 0
        for i, coef in enumerate(polynomial):
            result += coef * (x ** i)
        return result

    return function


def partial_derivative(f, coordinates0, coordinates1, n):
    copy = coordinates0.copy()
    copy[n] = coordinates1[n]
    return (f(copy) - f(coordinates0)) / (coordinates1[n] - coordinates0[n])


def newton(f, g, coor0, epsilon):
    """the newton method for 2 functions simultaneously"""
    cur = [epsilon*1000 + x for x in coor0]
    last = coor0
    while abs((cur[0]**2 + cur[1]**2) - (last[0]**2 + last[1]**2)) >= epsilon:
        delta = [epsilon + x for x in last]
        flast = f(cur)
        glast = g(cur)
        fx = partial_derivative(f, cur, delta, 0)
        fy = partial_derivative(f, cur, delta, 1)
        gx = partial_derivative(g, cur, delta, 0)
        gy = partial_derivative(f, cur, delta, 1)
        denom = (fx*gy) - (gx*fy)
        tag = [((flast*gy)-(glast*fy))/denom, ((glast*fx)-(flast*gx))/denom]
        last = cur
        cur = [cur[i] - tag[i] for i in range(2)]
    return cur


def row_reduction(A, c):
    # add pivoting!!!
    """ takes a nXn matrix and a vector c and returns the solutions in order"""
    matrix = np.append(A, c, axis=1)
    (rows, cols) = np.shape(matrix)
    """reduce down:"""
    for pivot in range(rows):
        a = matrix[pivot, pivot]
        copy = matrix[pivot].copy()
        matrix[pivot] = [x/a for x in copy]
        for row in range(pivot + 1, rows):
            b = matrix[row, pivot]
            matrix[row] = [elem - piv * b for elem, piv in zip(matrix[row], matrix[pivot])]
    """reduce up:"""
    for pivot in range(rows - 1, -1, -1):
        for row in range(pivot - 1, -1, -1):
            d = matrix[row, pivot]
            matrix[row] = [elem - piv * d for elem, piv in zip(matrix[row], matrix[pivot])]
    return np.array([r[-1] for r in matrix], dtype=float)


def crout(A):
    n, m = np.shape(A)
    if n != m:
        return "your matrix is not square!"
    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float)
    for i in range(n):
        L[i, i] = 1.0
    for col in range(n):
        for row in range(n):
            s = 0.0
            if row <= col:
                for k in range(0, row):
                    s += L[row, k] * U[k, col]
                U[row, col] = A[row, col] - s
            else:
                for k in range(0, col):
                    s += L[row, k] * U[k, col]
                L[row, col] = (A[row, col] - s) / U[col, col]
    return L, U


def down(L, c):
    y = []
    for i in range(len(c)):
        s = 0
        for j in range(i):
            s += L[i, j] * y[j]
        y.append((c[i] - s) / L[i, i])
    return y


def up(U, y):
    x = np.zeros(len(y), dtype=float)
    n = len(y)
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i+1, n):
            s += U[i, j] * x[j]
        x[i] = (y[i] - s) / U[i, i]
    return x


def LU_decomposition(A, c):
    L, U = crout(A)
    y = down(L, c)
    x = up(U, y)
    return x


def gauss_seidel(A, c, epsilon):
    ind = 0
    n, m = np.shape(A)
    x = np.zeros_like(c, dtype=np.double)
    if n != m:
        return "your matrix is not square!"
    convergence = np.zeros(n, dtype=bool)
    while ind<5: #not all(convergence):
        for i in range(n):
            ind += 1
            s = 0
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            """for j in range(n):
                if j != i:
                    a = A[i, j]
                    b = x[j]
                    print(f"a = {a}, b = {b}")
                    s += a*b
                    print(s)"""
            last = x[i]
            cur = (c[i] - s) / A[i][i]
            print(last, cur)
            x[i] = cur
            convergence[i] = abs(cur - last) <= epsilon
    return x


def jacobi(A, b, epsilon, iterations):
    n, m = np.shape(A)
    if n != m:
        return "matrix must be square"
    x = np.zeros_like(b)

    for k in range(iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        if all(abs(a - c) < epsilon for a, c in zip(x, x_new)):
            return x
        x = x_new
    return f"could not converge within the allowed iterations, the solutions found: {x}"

