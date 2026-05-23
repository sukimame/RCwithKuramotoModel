import numpy as np
import matplotlib.pyplot as plt

def adjust_spectral_radius(w, target_radius=0.95):
    eigenvalues = np.linalg.eigvals(w)
    spectral_radius = np.max(np.abs(eigenvalues))
    scaling_factor = target_radius / spectral_radius
    adjusted_w = w * scaling_factor
    
    return adjusted_w

class KURAMOTO:
    def __init__(self, n):
        self.lam = 6.6
        self.epsilon = 0.01
        self.beta = 0
        self.dt = 0.01
        self.n = n

        rng = np.random.default_rng()
        self.omega = 2 * np.pi * rng.standard_normal(n) * 0.5
        self.theta = 2 * np.pi * rng.standard_normal(n)
        self.k = rng.standard_normal((n, n))*0.5 + 0.5

        self.diff = np.zeros((n, n))

    def updateDiff(self):
        # theta_j (列) - theta_i (行) にすることで、sin(theta_j - theta_i) になる
        self.diff = self.theta[np.newaxis, :] - self.theta[:, np.newaxis]

    def updateTheta(self, u):
        d_theta = self.omega + self.lam * np.sum(self.k * np.sin(self.diff + u), axis=1) / self.n
        self.theta = self.theta + self.dt * d_theta
    
    def updateK(self):
        self.d_k = -self.epsilon * np.sin(self.diff + self.beta)
        self.k = self.k + self.dt * self.d_k
        adjust_spectral_radius(self.k, target_radius=2.6)

    
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
    
    def orderParam(self):
        x = np.mean(np.sin(self.theta))
        y = np.mean(np.cos(self.theta))
        return x, y

if __name__ == "__main__":
    km = KURAMOTO(100)
    km.visualizeTheta()

