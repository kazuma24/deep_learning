import sys, os
sys.path.append(os.pardir)
from common.functions import *
import numpy as np

class TwoLayerNet:
    
    # input_size=入力層のニューロン数 output_size=隠れ層のニューロン数 output_size=隠れ層のニューロン数
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        
        # W=重み b=バイアス
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        # 1層目の重み付け
        a1 = np.dot(x, W1) + b1
        # 1層目の出力(活性化関数にはシグモイド関数を使用)
        z1 = sigmoid(a1)
        
        #2層目の重み付け
        a2 = np.dot(z1, W2) + b2
        # 出力層 ソフトマックス関数を使用
        y = softmax(a2)
        
        return y
    
    # 損失関数 x:入力データ t:教師データ
    def loss(self, x, t):
        # 推論
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        
        # 推論
        y = self.predict(x)
        # 推論結果　最大index
        y = np.argmax(y, axis=1)
        # 教師データ 最大index
        t = np.argmax(y, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # 重みの勾配を求める x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
    
    # def gradient(self, x, t):
    #     W1, W2 = self.params['W1'], self.params['W2']
    #     b1, b2 = self.params['b1'], self.params['b2']
    #     grads = {}
        
    #     batch_num = x.shape[0]
        
    #     # forward
    #     a1 = np.dot(x, W1) + b1
    #     z1 = sigmoid(a1)
    #     a2 = np.dot(z1, W2) + b2
    #     y = softmax(a2)
        
    #     # backward
    #     dy = (y - t) / batch_num
    #     grads['W2'] = np.dot(z1.T, dy)
    #     grads['b2'] = np.sum(dy, axis=0)
        
    #     dz1 = np.dot(dy, W2.T)
    #     da1 = sigmoid_grad(a1) * dz1
    #     grads['W1'] = np.dot(x.T, da1)
    #     grads['b1'] = np.sum(da1, axis=0)

    #     return grads
        
        
    
    
        
        
        