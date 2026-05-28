from kuramoto import KURAMOTO
import numpy as np
import matplotlib.pyplot as plt

def almost_periodic(t):
    return np.sin(t) + np.sin(np.sqrt(2)*t)

class KURAMOTO_RC:
    def __init__(self, n=500):
        self.wout = np.ones(2*n+1)
        self.model = KURAMOTO(n, k=0.65, dt=0.01, alpha=0.01)
        self.n = n

    def batch_update(self, inputs):
        xs = np.zeros((inputs.shape[-1], self.n*2+1))
        xs[:, 0] = 1
        for i, u in enumerate(inputs):
            self.model.updateTheta(u)
            theta__ = self.model.theta[np.newaxis, :]*np.arange(1, self.n+1).reshape((self.n, 1))
            xs[i, 1:self.n+1] = np.mean(np.sin(theta__), axis=1)
            xs[i, self.n+1:] = np.mean(np.cos(theta__), axis=1)

        return xs

    def ridge(self, xs, ts):
        #self.wout = np.linalg.pinv(xs) @ ts
        self.wout = np.linalg.pinv(xs.T @ xs + 1e-2 * np.eye(xs.shape[1])) @ xs.T @ ts
    
    def autonomous_update(self, u):
        self.model.updateTheta(u)
        theta__ = self.model.theta[np.newaxis, :]*np.arange(1, self.n+1).reshape((self.n, 1))
        x = np.zeros(self.n*2+1)
        x[0] = 1
        x[1:self.n+1] = np.mean(np.sin(theta__), axis=1)
        x[self.n+1:] = np.mean(np.cos(theta__), axis=1)

        return x

if __name__ == "__main__":
    rc = KURAMOTO_RC()

    dt = 0.01

    washout = 1
    train = 1
    test = 1
    timeSec = washout + train + test # 240s for training, 32s for training, 32s for testing
    time = np.arange(0, timeSec, dt)

    inputs = almost_periodic(time)
    rc.batch_update(inputs[:int(washout/dt)])

    ts = almost_periodic(time + 100)
    xs = rc.batch_update(inputs[int(washout/dt):int((washout+train)/dt)])
    rc.ridge(xs, ts[int((washout)/dt):int((washout+train)/dt)])

    pred = rc.batch_update(inputs[int((washout+train)/dt):]) @ rc.wout

    #pred = np.zeros(test)

    acc = np.sum((ts[int((washout+train)/dt):] - pred)**2) / pred.shape[0]

    print("MSE", acc)

    plt.plot(pred)
    plt.plot(ts[int((washout+train)/dt):int((washout+train+test)/dt)+pred.shape[0]])
    plt.show()
