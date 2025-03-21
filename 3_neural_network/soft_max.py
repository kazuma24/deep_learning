import numpy as np

def softmax( a ):
    
    max_num = np.max( a )
    exp_a = np.exp(a - max_num)
    exp_sum = np.sum( exp_a )
    y = exp_a / exp_sum
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax( a )
print(y)
print(np.sum(y))

