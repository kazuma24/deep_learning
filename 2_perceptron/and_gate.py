# conding: utf-8
import numpy as np

# パーセプトロン形式のANDゲート関数(バイアス利用)
# x1 = 入力信号1
# x2 = 入力信号2
def AND(x1, x2):
    
    input = np.array([x1,x2])
    weight = np.array([0.5,0.5])
    
    # バイアス(信号発火のしやすさ->大きければ1になりやすい)
    b = -0.7
    
    # 入力信号+重み
    tmp = np.sum(input*weight) + b
    
    if tmp > 0:
        return 1
    elif tmp <= 0:
        return 0

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = AND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))