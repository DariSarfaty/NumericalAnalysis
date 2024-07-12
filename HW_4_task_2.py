import Tools
import numpy as np

def f(x):
    return x*np.e**(x*2)


interval = [0, 4]
digits = 7
epsilon = 10 ** (-1*digits)
real = 5216.926477323024480801286174
rounded = round(real, digits)

# ran this to find out that the best res achieveable in reasonable time using the trapezoidal method is 0.01 in 2049 steps
"""for i in range(1, 100000):
    trap = Tools.trap(f, interval, i)
    if round(trap, 2) == round(real, 2):
        break
print(i, trap)"""

steps = 2049
trap = round(Tools.trap(f, interval, steps), digits)


# ran this to find out that the steps required for the wanted res using the simpson 1/3 method is 6039 steps
"""for i in range(steps, 100000, 2):
    calc = Tools.simpson13(f, interval, i)
    if round(calc, digits) == rounded:
        break
print(i, calc)"""

steps = 6039
simpson = round(Tools.simpson13(f, interval, steps), digits)


# using romberg's method took 7 trapezoidal evaluations (the last one with 64 steps) to calculate the integral with the required resolution.

romberg = round(Tools.romberg(f, interval, Tools.trap, epsilon), digits)


quad = round(Tools.quad10(f, interval), digits)

print(f"calculated the integral using 4 methods: \ntrapezoidal: {trap}\nsimpson: {simpson}\nromberg: {romberg}"
      f"\nquadrature using 10 points: {quad}\nthe real value is: {rounded}")