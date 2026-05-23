import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from kuramoto import KURAMOTO

km = KURAMOTO(100)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.legend()

def update(frame):
    km.updateDiff()
    km.updateTheta(u=0)

    if frame < 1000:
        km.updateK()
        print(np.mean(km.k))

    plt.cla()
    ax.scatter(np.sin(km.theta), np.cos(km.theta), color='red')
    ax.plot(np.sin(np.linspace(0, 2*np.pi, 10000)), np.cos(np.linspace(0, 2*np.pi, 10000)), c="gray", alpha=0.7, zorder=1)

    x, y = km.orderParam()
    ax.scatter(x, y, c="blue", zorder=2)

ani = animation.FuncAnimation(
    fig, update,
    frames=1000,
    interval=10,
    blit=False
)
plt.show()
