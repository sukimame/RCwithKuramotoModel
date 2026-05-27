import numpy as np

a = np.arange(5)
"""
b = np.tile(a, (5, 1))
c = a.reshape((5, 1))
c = np.tile(c, (1, 5))

print(c)

# b: 横に伸びる行列, c: 縦に伸びる行列
b, c = np.meshgrid(a, a)

print(b, c)

print(b-c)
d = np.sum(b - c, axis=1)

print(d)
"""

a = np.arange(1,4)

print(a)

b = np.array([1, 10, 100])







print(a[np.newaxis, :]*b)