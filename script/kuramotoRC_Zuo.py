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

class KURAMOTO_RC_ZUO:
    def __init__(
            self, 
            n, 
            dt=1.0, 
            alpha=1.0, 
            lambda_=4.0, 
            radius=1.1,
            epsilon=0.1,
            beta=np.pi/2
            ): 
        
        self.dt = dt
        self.n = n
        self.alpha = alpha
        self.lambda_ = lambda_
        self.radius = radius
        self.epsilon = epsilon
        self.beta = beta

        rng = np.random.default_rng()
        #self.omega = rng.uniform(-0.2, 0.6, n)
        self.omega = rng.normal(0.5,0.5,n)
        self.theta = np.zeros(n)

        self.wout = np.ones(2*n+1)
        
        self.mask = (np.random.rand(n, n) < 1).astype(float)
        self.mask *= (1 - np.eye(n))
        #self.k = rng.uniform(0, 1, (n, n)) * self.mask
        self.k = np.ones((n, n))*0.7 * self.mask

    def updateDiff(self):
        # theta_j (列) - theta_i (行) にすることで、sin(theta_j - theta_i) になる
        return self.theta[np.newaxis, :] - self.theta[:, np.newaxis]

    def updateTheta(self, u=0):
        d_theta =  self.omega + self.lambda_ * np.sum(self.k * np.sin(self.updateDiff() + self.alpha*u), axis=1)
        self.theta = (self.theta + self.dt * d_theta) #% 2*np.pi

    def updateK(self):
        self.d_k = -self.epsilon * np.sin(self.updateDiff() + self.beta)
        self.k = self.k + self.dt * self.d_k
        self.k[self.k<-1] = -1
        self.k[self.k>1] = 1
        self.k *= self.mask
        self.k = adjust_spectral_radius(self.k, target_radius=self.radius)
        #print(np.mean(self.k), np.std(self.k))

    def washout(self, inputs):
        for i, u in enumerate(inputs):
            self.updateTheta(u)
            #self.updateK()

    def batch_update(self, inputs):
        xs = np.zeros((inputs.shape[-1], 2*self.n+1))
        xs[:, 0] = 1

        op = []
        for i, u in enumerate(inputs):
            self.updateTheta(u)
            xs[i, 1:self.n+1] = np.sin(self.theta)
            xs[i, self.n+1:] = np.cos(self.theta)
            x, y = self.orderParam(1)
            #print(np.sqrt(x**2+y**2))
            op.append(np.sqrt(x**2+y**2))

        return xs, op
    
    def ridge(self, xs, ts):
        self.wout = np.linalg.pinv(xs.T @ xs + 1e-3 * np.eye(xs.shape[1])) @ xs.T @ ts

    def orderParam(self, n=1):
        x = np.mean(np.sin(self.theta * n))
        y = np.mean(np.cos(self.theta * n))
        return x, y


if __name__=="__main__":

    dt = 1
    rc = KURAMOTO_RC_ZUO(
            n=100,
            dt=dt,
            alpha=0.01,
            lambda_=0.01,
            radius=1.5,
            epsilon=0.1,
            beta=np.pi/2)

    washout = 100
    train = 900
    test = 500
    timeSec = washout + train + test 
    time = np.arange(0, timeSec, dt)

    data, _ = create_narma10_dataset(train_samples=int((timeSec)/dt), test_samples=1, seed=42)

    rc.washout(data['input'][:int(washout/dt)])

    xs, op = rc.batch_update(data['input'][int(washout/dt):int((washout+train)/dt)])
    rc.ridge(xs, data['target'][int(washout/dt):int((washout+train)/dt)])

    pred = rc.batch_update(data['input'][int((washout+train)/dt):]) @ rc.wout

    mse = np.sum((data['target'][int((washout+train)/dt):] - pred)**2) / pred.shape[0]
    rmse = np.sqrt(mse)

    target_std = np.std(data['target'][int((washout+train)/dt):])
    nrmse = rmse / target_std

    print(np.mean(rc.theta % 2*np.pi), np.std(rc.theta % 2*np.pi))

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"NRMSE: {nrmse:.4f}")    

    #plt.plot(pred)
    #plt.plot(data['target'][int((washout+train)/dt):], alpha=0.5)
    plt.plot(op)
    plt.legend()
    plt.show()