import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error, numerical_gradient

class simpleNet:
    
    def __init__(self):
        # 重みパラメータ
        self.W = np.random.randn(2,3)
        # self.W = np.array([
        #     [-0.50613378,  0.49075257, -0.50350052],
        #     [ 0.77562224,  1.44039291, 4.90153086]
        # ])
    
    # x= 入力データ
    def predict(self, x):
        return np.dot(x, self.W)
    
    # x=入力データ t=正解ラベル
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss
    
net = simpleNet()
print('重みパラメータ', net.W)

# 入力データx
x = np.array([0.6, 0.9])
print('入力データ', x)
# 推論
p = net.predict(x)
print('推論結果', p)
print('最大値インデックス', np.argmax(p))

t = np.array([0,0,1])
print('正解ラベル', t)

loss = net.loss(x, t)
print('損失関数', loss)

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print('dW', dW)

