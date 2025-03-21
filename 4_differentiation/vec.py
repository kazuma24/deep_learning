import numpy as np

# 行列の和
A = np.array([
    [1,2],
    [3,4]
])
B = np.array([
    [1,2],
    [3,4]
])
print(A+B)

# 行列の差
C = np.array([
    [1,2,3],
    [-1,2,1],
    [0,1,1]
])
D = np.array([
    [1,1,1],
    [1,2,1],
    [-1,1,1]
])
print(C-D)

# スカラー倍
E = np.array([
    [1,2,0],
    [4,-1, 3]
])
print(E*3)

# 積
print('-積-')
F = np.array([
    [1,3],
    [-3,4]
])
F2 = np.array([
    [1,9],
    [9,3]
])
print(F @ F2)
G = np.array([
    [1,0],
    [4,1],
    [-1,2]
])
G1 = np.array([
    [1,3],
    [2,-1]
])
print(np.dot(G,G1))