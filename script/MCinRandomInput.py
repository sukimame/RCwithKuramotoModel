from kuramotoRC_Zuo import KURAMOTO_RC
import matplotlib.pyplot as plt
import numpy as np
from NARMA import create_narma10_dataset

"""
def lowhighSin(t):
    return np.sin(t) + np.sin(10*np.sqrt(2)*t)

inputs = lowhighSin(time)
ts = lowhighSin(time + 100)
"""

def RandomInput(n, delay):
    seed = 42
    np.random.seed(seed)

    # Generate random input uniformly distribu
    data = np.random.uniform(0, 0.5, n + delay)
    return data[:-delay], data[delay:]

rc = KURAMOTO_RC()

dt = 1

washout = 900
train = 500
test = 500
timeSec = washout + train + test # 240s for training, 32s for training, 32s for testing
time = np.arange(0, timeSec, dt)

seed = 24
np.random.seed(seed)
washout_data = np.random.uniform(0, 0.5, int(washout/dt))
x, y = RandomInput(int((train+test)/dt), 1)

rc.batch_update(washout_data)

xs = rc.batch_update(x[:int(train/dt)])
rc.ridge(xs, y[:int(train/dt)])

pred = rc.batch_update(x[int(train/dt):]) @ rc.wout

#pred = np.zeros(test)
mse = np.sum((y[int(train/dt):] - pred)**2) / pred.shape[0]
rmse = np.sqrt(mse)

target_std = np.std(y[int(train/dt):])
nrmse = rmse / target_std

print("MSE", mse)
print("RMSE", rmse)
print("NRMSE", nrmse)

plt.plot(pred, label="pred")
plt.plot(y[int(train/dt):], label="true")
plt.legend()
plt.show()
