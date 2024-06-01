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

