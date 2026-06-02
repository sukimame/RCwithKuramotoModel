import numpy as np
import matplotlib.pyplot as plt

class KURAMOTO:
    def __init__(self, n, k=0.637, dt=0.01, alpha=0.01):
        self.k = k  
        self.dt = dt
        self.n = n
        self.alpha = alpha

        rng = np.random.default_rng()
        #self.omega = np.random.normal(0.2, 0.2, n)
        self.omega = np.random.uniform(-0.3, 0.7, n)
        #self.omega = np.linspace(6.5, 8.5, n)
        self.theta = 2 * np.pi * rng.random(n)

    def updateDiff(self):
        # theta_j (列) - theta_i (行) にすることで、sin(theta_j - theta_i) になる
        return self.theta[np.newaxis, :] - self.theta[:, np.newaxis]

    def updateTheta(self, u=0):
        d_theta =  self.omega + self.k * np.sum(np.sin(self.updateDiff()+self.alpha*u), axis=1) / self.n
        self.theta = self.theta + self.dt * d_theta

    def updateK(self):
        self.d_k = -self.epsilon * np.sin(self.updateDiff() + self.beta)
        self.k = self.k + self.dt * self.d_k
        #adjust_spectral_radius(self.k, target_radius=2.6)
    
    def visualizeTheta(self):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.scatter(np.sin(self.theta), np.cos(self.theta), c = "r", zorder=2)
        ax.plot(np.sin(np.linspace(0, 2*np.pi, 10000)), np.cos(np.linspace(0, 2*np.pi, 10000)), c="gray", alpha=0.7, zorder=1)
        x, y = self.orderParam()
        ax.scatter(x, y, c="blue", zorder=2)
        plt.show()
    
    def orderParam(self, n=1):
        x = np.mean(np.sin(self.theta * n))
        y = np.mean(np.cos(self.theta * n))
        return x, y

if __name__ == "__main__":
    km = KURAMOTO(100)
    km.visualizeTheta()

