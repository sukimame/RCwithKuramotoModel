import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from NARMA import create_narma10_dataset

class KURAMOTORCEX:
    def __init__(
            self, 
            n=100, 
            dt=0.01, 
            alpha=0.01, 
            K = 0.7,
            s=1,
            gamma=1
        ): 
        self.n = n  
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_=1

        rng = np.random.default_rng()

        self.omega = rng.normal(0.2, 0.2, n)
        #self.omega = np.random.uniform(-0.3, 0.7, n)
        #self.omega = np.linspace(6.5, 8.5, n)
        self.theta = rng.uniform(0, 2*np.pi, n)
        self.d_theta = np.zeros(n)

        self.wout = np.ones(n+1)

        self.mask = (np.random.rand(n, n) < s).astype(float)
        self.mask *= (1 - np.eye(n))
        #self.k = rng.uniform(0, 1, (n, n)) * self.mask
        self.k = np.ones((n, n)) * K/n * self.mask

    def updateDiff(self):
        # theta_j (列) - theta_i (行) にすることで、sin(theta_j - theta_i) になる
        return self.theta[np.newaxis, :] - self.theta[:, np.newaxis]
    
    def RK(self):
        

    def updateTheta(self, u=0):
        self.d_theta = self.omega + self.lambda_ * np.sum(self.k * np.sin(self.updateDiff() + self.alpha*u), axis=1)
        self.theta = (self.gamma * self.theta + self.dt * self.d_theta)# % 2*np.pi
    
    def orderParam(self, n=1):
        x = np.mean(np.sin(self.theta * n))
        y = np.mean(np.cos(self.theta * n))
        return x, y

    def washout(self, inputs):
        x, y = self.orderParam(1)
        print(np.sqrt(x**2+y**2))
        for i, u in enumerate(inputs):
            self.updateTheta(u)

    def batch_update(self, inputs):
        xs = np.zeros((inputs.shape[-1], self.n+1))
        xs[:, 0] = 1

        for i, u in enumerate(inputs):
            self.updateTheta(u)
            #xs[i, 1:self.n+1] = np.sin(self.theta)
            #xs[i, self.n+1:] = np.cos(self.theta)
            xs[i, 1:] = self.d_theta * int(1/self.dt) #- self.omega
        x, y = self.orderParam(1)
        print(np.sqrt(x**2+y**2))
          
        return xs
    
    def ridge(self, xs, ts):
        self.wout = np.linalg.pinv(xs.T @ xs + 1e-7 * np.eye(xs.shape[1])) @ xs.T @ ts
    
    """
    def ridge(self, xs, ts):
        #self.wout = np.linalg.pinv(xs) @ ts
        self.wout = np.linalg.pinv(xs.T @ xs + 1e-2 * np.eye(xs.shape[1])) @ xs.T @ ts
    """

    def orderParam(self, n=1):
        x = np.mean(np.sin(self.theta * n))
        y = np.mean(np.cos(self.theta * n))
        return x, y

if __name__=="__main__":
    dt = 0.01
    rc = KURAMOTORCEX(
            n=100,
            dt=dt,
            alpha=0.01,
            K=0.6,
            s=1
            )

    washout = 100
    train = 5
    test = 200
    timeSec = washout + train + test 
    time = np.arange(0, timeSec, dt)

    data, _ = create_narma10_dataset(train_samples=int((timeSec)/dt), test_samples=1, seed=42)

    rc.washout(data['input'][:int(washout/dt)])

    xs = rc.batch_update(data['input'][int(washout/dt):int((washout+train)/dt)])
    rc.ridge(xs, data['target'][int(washout/dt):int((washout+train)/dt)])

    pred = rc.batch_update(data['input'][int((washout+train)/dt):])
    pred = pred @ rc.wout

    mse = np.sum((data['target'][int((washout+train)/dt):] - pred)**2) / pred.shape[0]
    rmse = np.sqrt(mse)

    target_std = np.std(data['target'][int((washout+train)/dt):])
    nrmse = rmse / target_std

    print(np.mean(rc.theta % 2*np.pi), np.std(rc.theta % 2*np.pi))
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"NRMSE: {nrmse:.4f}")

    print("mean, pred and target", np.mean(pred), np.mean(data['target'][int((washout+train)/dt):]))
    print("std, pred and target", np.std(pred), target_std)

    print(np.median(rc.wout), np.std(rc.wout))


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(pred)
    ax1.plot(data['target'][int((washout+train)/dt):], alpha=0.5)

    plt.legend()
    plt.show()

