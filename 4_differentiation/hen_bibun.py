import numpy as np


# 数値微分(中心差分)
def numeric_diff(f, x):
    
    # 変化量
    h = 1e-4
    
    # 微分
    y = (f(x + h) - f(x-h)) / (2*h)
    
    return y

# f(x0, x1) = x0**2 + x1**2
# x0=3, x1=4の時のx0に対する偏微分を求める
result1 = numeric_diff(
    lambda x0: x0**2 + 4.0**2,
    x=3.0
)

print(result1)

# x0=3, x1=4の時のx1に対する偏微分を求める
result2 = numeric_diff(
    lambda x1: 3**2 + x1**2,
    x=4.0
)

print(result2)