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




