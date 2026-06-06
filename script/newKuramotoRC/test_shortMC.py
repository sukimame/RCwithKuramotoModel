from kuramotoRCEXRK4 import KURAMOTORCEX
import matplotlib.pyplot as plt
import numpy as np

def RandomInput(n, delay):
    np.random.seed(42)
    data = np.random.uniform(0, 0.6, n)
    return data[delay:], data[:-delay]

dt = 0.01
rc = KURAMOTORCEX(
        n=100,
        dt=dt,
        alpha=0.01,
        K=1,
        s=1
        )
print(rc.k)

washout = 100
train = 5
test = 5
timeSec = washout + train + test 
time = np.arange(0, timeSec, dt)

data, label = RandomInput(int(timeSec/dt), 1)

rc.washout(data[:int(washout/dt)])

xs = rc.batch_update(data[int(washout/dt):int((washout+train)/dt)])
rc.ridge(xs, label[int(washout/dt):int((washout+train)/dt)])

pred = rc.batch_update(data[int((washout+train)/dt):])
pred = pred @ rc.wout

mse = np.sum((label[int((washout+train)/dt):] - pred)**2) / pred.shape[0]
rmse = np.sqrt(mse)

target_std = np.std(label[int((washout+train)/dt):])
nrmse = rmse / target_std

print("sd of theta", np.std(rc.theta))
print(f"NRMSE: {nrmse:.4f}")    


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(pred, label="pred")
ax1.plot(label[int((washout+train)/dt):], alpha=0.5, label="target" )

plt.legend()
plt.show()
