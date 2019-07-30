import numpy as np

a = np.ones(shape=(10, 256, 256, 3))

print(a.shape)

b = a[0:5, 0:224, 0:224]

print(b.shape)
