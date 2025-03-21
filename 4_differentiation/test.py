import numpy as np
import matplotlib.pyplot as plt

# 数値微分(中心差分)
def numerical_diff(f, x):
    h = 1e-4
    y = (f(x + h) - f(x - h)) / (h*2)
    return y

# y = 0.01xe2 + 0.1x
def function_1(x):
    y = 0.01*x**2 + 0.1*x
    return y

# f(x0, x1) = x0**2 + x1**2
def function_tmp(x0):
    return x0**x0 + 4.0**2

result = numerical_diff(function_tmp, x=3.0)
print(result)

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")

# tf = tangent_line(function_1, 5)
# y2 = tf(x)

# plt.plot(x, y)
# plt.plot(x, y2)
# plt.show()