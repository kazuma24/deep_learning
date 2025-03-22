import numpy as np

def identity_function(x):
    return x

# ステップ関数
def step_function(x):
    return np.array(x > 0, dtype=int)

# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(x))

# ReLu関数
def relu(x):
    return np.maximum(0, x)

# ソフトマックス関数
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# 損失関数 (交差エントロピー法)
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# 勾配法
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


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)