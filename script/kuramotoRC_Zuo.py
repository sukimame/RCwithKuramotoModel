from kuramoto import KURAMOTO
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from NARMA import create_narma10_dataset

def adjust_spectral_radius(w, target_radius=1.1):
    eigenvalues = np.linalg.eigvals(w)
    spectral_radius = np.max(np.abs(eigenvalues))
    scaling_factor = target_radius / spectral_radius
    adjusted_w = w * scaling_factor
    
    return adjusted_w

class KURAMOTO_RC:
    def __init__(self, n=100, dt=1.0):
        self.wout = np.ones(n+1)
        self.model = KURAMOTO(n, k=0.65, dt=dt, alpha=1.0)
        self.n = n

        #接続重み　論文中でスパースさが精度にそこまで影響していないことが示されている
        self.k = np.random.uniform(-1, 1, (n, n))
        self.k = self.k * (np.ones((n, n)) - np.eye(n))
        self.k = adjust_spectral_radius(self.k)

        self.omega = self.model.omega
        self.dt = self.model.dt
        self.lambda_ = 4.0
    
    def updateTheta(self, u=0):
        d_theta =  self.omega + self.lambda_ * np.sum(self.k * np.sin(self.model.updateDiff() + self.model.alpha*u), axis=1)
        self.model.theta = (self.model.theta + self.dt * d_theta) % (2*np.pi)

    def batch_update(self, inputs):
        xs = np.zeros((inputs.shape[-1], self.n+1))
        xs[:, 0] = 1
        for i, u in enumerate(inputs):
            self.updateTheta(u)
            xs[i, 1:] = np.sin(self.model.theta)

        return xs

    def ridge(self, xs, ts):
        self.wout = np.linalg.pinv(xs.T @ xs + 1e-3 * np.eye(xs.shape[1])) @ xs.T @ ts

if __name__=="__main__":

    rc = KURAMOTO_RC(n=100, dt=1)
    dt = 1

    washout = 100
    train = 900
    test = 500
    timeSec = washout + train + test 
    time = np.arange(0, timeSec, dt)

    train_data, test_data = create_narma10_dataset(train_samples=int((washout+train)/dt), test_samples=int(test/dt), seed=42)

    rc.batch_update(train_data['input'][:int(washout/dt)])

    xs = rc.batch_update(train_data['input'][int(washout/dt):int((washout+train)/dt)])
    rc.ridge(xs, train_data['target'][int(washout/dt):int((washout+train)/dt)])

    pred = rc.batch_update(test_data['input']) @ rc.wout

    #pred = np.zeros(test)
    mse = np.sum((test_data['target'] - pred)**2) / pred.shape[0]
    rmse = np.sqrt(mse)

    target_std = np.std(test_data['target'])
    nrmse = rmse / target_std

    print("MSE", mse)
    print("RMSE", rmse)
    print("NRMSE", nrmse)

    plt.plot(pred)
    plt.plot(test_data['target'])
    plt.show()
