import numpy as np

# 定义输入数组和卷积核
a = np.array([1, 1, 2, 3, 4, 5, 5])
v = np.array([1, 2, 3])

# 计算线性卷积
res = np.convolve(a, v, mode='full')

print(res)


















# def my_decorator(func):
#     def wrapper(*args, **kwargs):
#         print("Before call")
#         result = func(*args, **kwargs)
#         print("After call")
#         return result
#     return wrapper

# @my_decorator
# def add(a, b):
#     print('in func')
#     return a + b

# print(add(1, 3))