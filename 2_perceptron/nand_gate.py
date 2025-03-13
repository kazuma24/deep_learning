import numpy as np


def NAND(x1,x2):
    
    i = np.array([-0.5,-0.5])
    w = np.array([x1,x2])
    b = 0.7
    
    # 入力*重み + バイアス
    tmp = np.sum(i*w) + b
    if tmp > 0:
        return 1
    elif tmp <= 0:
        return 0

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = NAND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))