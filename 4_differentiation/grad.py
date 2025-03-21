import numpy as np

def numeric_diff_grad(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        
        tmp_v = x[idx]
        # f(x+h)の計算
        x[idx] = tmp_v+h
        fxh1 = f(x)
        # f(x-h)の計算
        x[idx] = tmp_v-h
        fxh2 = f(x)
        # 微分
        result = (fxh1 - fxh2) / (2*h)
        grad[idx] = result
        # 値をもとに戻す
        x[idx] = tmp_v
    
    return grad

# f(x0, x1) = x0**2 + x1**2
def function_1(x):
    return np.sum(x**2)

ary1 = np.array([3.0, 4.0])
result1 = numeric_diff_grad(
    function_1,
    ary1
)

ary2 = np.array([0.0, 2.0])
result2 = numeric_diff_grad(
    function_1,
    ary2
)

ary3 = np.array([3.0, 0.0])
result3 = numeric_diff_grad(
    function_1,
    ary3
)

print(result1, result2, result3)


        