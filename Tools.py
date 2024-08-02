import numpy as np
import matplotlib.pyplot as plt


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
        """partial pivoting:"""
        matrix = pivoting_matrix(matrix, pivot)
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
    x = []
    L, U = crout(A)
    for row in zip(*c):
        y = down(L, row)
        x.append(up(U, y))
    return x


def jacobi(A, b, epsilon=0.000001, iterations=100):
    """make the matrix diagonally dominant:"""
    A, b = pivoting_A_b(A, b, 0)
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


def gauss_seidel(A, b, epsilon=0.000001, iterations=100):
    """make the matrix diagonally dominant:"""
    A, b = pivoting_A_b(A, b, 0)
    n, m = np.shape(A)
    if n != m:
        return "matrix must be square"
    x = np.zeros_like(b)

    for k in range(iterations):
        x_new = x.copy()
        for i in range(n):
            s = sum(A[i, j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        if all(abs(a - c) < epsilon for a, c in zip(x, x_new)):
            return x
        x = x_new
    return f"could not converge within the allowed iterations, the solutions found: {x}"

def pivoting_A_b(A, b, pivot):
    matrix = np.append(A, b, axis=1)
    n, m = np.shape(A)
    for i in range(pivot,n-1):
        values = matrix[:, i]
        row = np.argmax(values)
        matrix[[i,row]] = matrix[[row,i]]
    return matrix[:, 0:n], matrix[:,n]

def pivoting_matrix(matrix, pivot):
    n, m = np.shape(matrix)
    for i in range(pivot,n-1):
        values = matrix[:, i]
        row = np.argmax(values)
        matrix[[i,row]] = matrix[[row,i]]
    return matrix


"""disclaimer: copied from 
https://stackoverflow.com/questions/37565793/how-to-let-the-user-select-an-input-from-a-finite-list
 :)"""
def selectFromDict(options, name):
    index = 0
    indexValidList = []
    print('Select a ' + name + ':')
    for optionName in options:
        index = index + 1
        indexValidList.extend([options[optionName]])
        print(str(index) + ') ' + optionName)
    inputValid = False
    while not inputValid:
        inputRaw = input(name + ': ')
        inputNo = int(inputRaw) - 1
        if inputNo > -1 and inputNo < len(indexValidList):
            selected = indexValidList[inputNo]
            inputValid = True
            break
        else:
            print('Please select a valid ' + name + ' number')
    return selected

def transpose(A):
    n,m = np.shape(A)
    transposed = np.zeros((m, n))
    for i in range(n):
        for j in range (m):
            transposed[j, i] = A[i, j]
    return transposed

def matrix_mult(A,B):
    result = [[sum(a * b for a, b in zip(A_row, B_col))
               for B_col in zip(*B)]
              for A_row in A]
    return result


def lagrange(data):
    def interpolation(point):
        P = 0
        for i in range(len(data)):
            L = 1
            xi = data[i, 0]
            yi = data[i, 1]
            for j in range(len(data)):
                if j != i:
                    xj = data[j, 0]
                    L *= (point - xj)/(xi - xj)
            P += L*yi
        return P
    return interpolation


def cubic_spline(data, res):
    x = [point[0] for point in data]
    y = [point[1] for point in data]
    n = len(x) - 1
    h = [x[i + 1] - x[i] for i in range(n)]

    alpha = np.zeros(n + 1)
    for i in range(1, n):
        alpha[i] = (3 / h[i] * (y[i + 1] - y[i])) - (3 / h[i - 1] * (y[i] - y[i - 1]))

    l = np.ones(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)

    for i in range(1, n):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    b = [0] * n
    c = [0] * (n + 1)
    d = [0] * n
    a = y[:-1]

    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    m = n - 1

    for i in range(n):
        ys = []
        myx = np.arange(x[i], x[i+1] + res, res)
        for s in myx:
            s = s - x[i]
            ys.append(a[i] + b[i] * s + c[i] * s ** 2 + d[i] * s ** 3)
        plt.plot(myx, ys, "b",lw=1)

    plt.plot(x, y, "r.")
    plt.show()
    return


def trap(func, interval, steps):
    x0 = interval[0]
    x1 = interval[1]
    h = (x1 - x0)/steps
    sum = func(x0) + func(x1)
    for i in range(1, steps):
        sum += 2*func(x0 + i*h)
    return sum *h / 2

def richardson(func, interval, min_steps):
    return 4/3*trap(func,interval,min_steps *2) - trap(func, interval, min_steps)/3


def simpson13(func, interval, steps):
    if steps%2 == 0:
        steps += 1
    x0 = interval[0]
    x1 = interval[1]
    h = (x1 - x0)/steps
    sum = func(x0) + func(x1)
    for i in range(1, steps):
        if i%2 == 0:
            factor = 4
        else:
            factor = 2
        sum += factor * func(x0 + i*h)
    result = sum * h / 3
    return result


def romberg_step(array, eval):
    myarr = [eval]
    for j in range(len(array)):
        k = j + 1
        a = 4**k
        myarr.append((a * myarr[j] - array[j]) / (a - 1))
    return myarr


def romberg(func, interval, method, epsilon):
    i = 0
    arr = []
    while i <= 1 or abs(arr[-1] - arr[-2]) > epsilon:
        steps = 2 ** i
        eval = method(func, interval, steps)
        arr = romberg_step(arr, eval)
        i += 1
    return arr[-1]


def quad10(func, interval):
    X = [-0.14887434, 0.14887434, -0.43339539, 0.43339539, -0.67940957, 0.67940957, -0.86506337, 0.86506337, -0.97390653, 0.97390653]
    C = [0.29552422, 0.29552422, 0.26926672, 0.26926672, 0.21908636, 0.21908636, 0.14945135, 0.14945135, 0.06667134, 0.06667134]
    a = interval[0]
    b = interval[1]
    A = (b - a) / 2
    B = (b + a) / 2

    def new_func(x):
        return A * func(A * x + B)

    sum = 0
    for x, c in zip(X, C):
        sum += c * new_func(x)
    return sum

def diff(x0, x1, fx0, fx1):
    return (fx1 - fx0)/(x1 - x0)

def func_diff(func, x, delta):
    x0 = x - delta
    x1 = x + delta
    fx0 = func(x0)
    fx1 = func(x1)
    return diff(x0, x1, fx0, fx1)

def romberg_diff(func, x, epsilon):
    i = 0
    arr = []
    while i <= 1 or abs(arr[-1] - arr[-2]) > epsilon:
        delta = 1 / (2 ** i)
        eval = func_diff(func, x, delta)
        arr = romberg_step(arr, eval)
        i += 1
    return arr[-1]


def euler(func, x0, y0, x, step):
    steps = int((x - x0)/step)
    for i in range(steps):
        y0 += func(x0, y0) * step
        x0 += step
    return y0


def RK2(func, x0, y0, x, step):
    steps = int((x - x0) / step)
    for i in range(steps):
        k1 = func(x0, y0)
        k2 = func(x0 + step, y0 + k1 * step)
        y0 += 0.5 * step * (k1 + k2)
        x0 += step
    return y0


def RK4(func, x0, y0, x, step):
    steps = int((x - x0) / step)
    for i in range(steps):
        k1 = func(x0, y0)
        k2 = func(x0 + 0.5*step, y0 + k1 * 0.5*step)
        k3 = func(x0 + 0.5*step, y0 + k2 * 0.5*step)
        k4 = func(x0 + step, y0 + k3 * step)
        y0 += step * (k1 + 2*k2 + 2*k3 + k4) / 6
        x0 += step
    return y0


def leapfrog(func, x0, v0, t, step):
    steps = int(t / step)
    for i in range(steps):
        a0 = func(x0)
        x0 += v0 * step + 0.5 * a0 * step * step
        a1 = func(x0)
        v0 += 0.5 * step * (a0 + a1)
    return x0, v0


def euler_2nd(func, x0, y0, v0, x, step):
    steps = int((x - x0) / step)
    for i in range(steps):
        y1 = y0 + step * v0
        v1 = v0 + step * func(x0, y0, v0)
        x0 += step
        y0 = y1
        v0 = v1
    return y0, v0



if __name__ == "__main__":
    def f(x, y):
        return 10 - 2*x

    print(RK4(f, 1, 1, 3, 0.1))