import numpy as np

k = np.ones((2, 2))
diff = np.arange(4).reshape((2, 2))

print(0.1 * np.sum(np.sin(diff), axis=1) / 4, np.sum(k * np.sin(diff), axis=1) * 0.25)