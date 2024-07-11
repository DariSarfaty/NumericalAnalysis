import Tools
import numpy as np


def f(x):
    return np.e**(-x*x)

interval = [0, 2]
steps = 10 #not 20 on purpose
digits = 7


print("trapezoid:" , round(Tools.trap(f,interval, steps*2), digits))
print("richardson:" , round(Tools.richardson(f,interval, steps), digits))
print("real:" , round(0.88208139076242167996748, digits))