"""disclaimer: I used chatGPT for this function only as I was not sure this was possible in python"""


def polynomial_to_function(polynomial):
    """
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
