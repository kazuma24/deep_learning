import numpy as np

# NumPy配列の生成
x = np.array([1,2,3])
print(type(x))

# NumPyの算術計算
x1 = np.array([1,2,3])
y1 = np.array([4,5,6])
# 同一要素の算術
print(x1+y1)
print(x1-y1)
print(x1*y1)
print(x1/y1)
# スカラ値を使用した算術
print(x1/2)

# NumPyのN次元配列
x2 = np.array([[1,2,3],[4,5,6]])
print(x2)
# 形状 -> shape
print(x2.shape)
# 型 -> dtype
print(x2.dtype)

y2 = np.array([[1,2,3],[4,5,6]])
print(x2+y2)

# 1次元配列へ平坦化
x3 = np.array([[1,2,3],[4,5,6]])
x4 = x3.flatten()
print(x4)