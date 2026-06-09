import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, np.pi*3, 10000)
y = np.sin(t) + np.sin(np.sqrt(2)*t)

plt.plot(t, y, "--", label="sin(x)+sin(sqrt(2)x)")
plt.scatter(t[::1000], y[::1000], marker="o", label="data point")
plt.legend()
plt.show()

print("\n")
for i in range(10):
    print(f"{t[1000*i]:.2f}", end=",")
