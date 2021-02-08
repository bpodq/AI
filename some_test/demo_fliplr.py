import numpy as np

A = np.random.randint(0, 2, [2, 2, 3])
B = np.fliplr(A)

print(A)  # 注意显示方式
print(B)  # 这样看是不行的
print('-------------')

print('fliplr: 对前两维构成的矩阵进行左右翻转')
print(A[:, :, 0])
print(B[:, :, 0])
print('-------------')

print(A[:, :, 1])
print(B[:, :, 1])
print('-------------')

print(A[:, :, 2])
print(B[:, :, 2])
print('-------------')

