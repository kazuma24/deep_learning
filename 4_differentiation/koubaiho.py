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
    return x[0]**2 + x[1]**2

# f=最適化したい関数 init_x = 初期値 lr=学習率 step_num=勾配法による繰り返し
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    
    x = init_x
    
    for i in range(step_num):
        
        grad = numeric_diff_grad(f, x)
        x -= lr * grad
        print(str(i), x)
    return x

init_x = np.array([-3.0, 4.0])
r = gradient_descent(
    function_1,
    init_x,
    0.1
)

print(r)

print(1e2)

        