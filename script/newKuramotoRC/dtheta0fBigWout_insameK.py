import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from NARMA import create_narma10_dataset
from MovingAvg import create_ma_dataset

from kuramotoRCEXRK4 import KURAMOTORCEX

dt = 0.01
rc = KURAMOTORCEX(
        n=100,
        dt=dt,
        alpha=0.1,
        K=3,
        s=1
        )

washout = 200
train = 5
test = 0.1
timeSec = washout + train + test 
time = np.arange(0, timeSec, dt)

#data, _ = create_narma10_dataset(train_samples=int((timeSec)/dt), test_samples=1, seed=24)
data, _ = create_ma_dataset(train_samples=int((timeSec)/dt), test_samples=1, window_size=10, seed=24)

rc.washout(data['input'][:int(washout/dt)])

xs = rc.batch_update(data['input'][int(washout/dt):int((washout+train)/dt)])
rc.ridge(xs, data['target'][int(washout/dt):int((washout+train)/dt)])

xs_pred = rc.batch_update(data['input'][int((washout+train)/dt):])
pred = xs_pred @ rc.wout

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

print(np.median(rc.wout), np.std(rc.wout), rc.wout)

"""
fig = plt.figure()
ax1 = fig.add_subplot(111)  
ax1.plot(data['input'][int((washout+train)/dt):], label="input")
ax1.plot(data['target'][int((washout+train)/dt):], label="output")

ax2 = ax1.twinx()
for i in np.arange(101)[rc.wout>0.5]:
        ax2.plot(xs_pred.T[i, :], alpha=0.2, c="red")

for i in np.arange(101)[rc.wout<-0.5]:
        ax2.plot(xs_pred.T[i, :], alpha=0.2, c="blue")

idx = (-0.5<=rc.wout) & (rc.wout<=0.5)
print(idx)
for i in np.arange(101)[idx]:
        ax2.plot(xs_pred.T[i, :], alpha=0.1, c="green")
plt.legend()
plt.show()
"""

fig = plt.figure()
ax1 = fig.add_subplot(111)  
ax1.plot(data['input'][int((washout+train)/dt):], label="input")
ax1.plot(data['target'][int((washout+train)/dt):], label="output")
ax1.plot(pred, label="pred")

ax2 = ax1.twinx()
for i in np.arange(rc.n+1)[rc.wout>0.5]:
        ax2.plot(xs_pred.T[i, :], alpha=0.2, c="red")

for i in np.arange(rc.n+1)[rc.wout<-0.5]:
        ax2.plot(xs_pred.T[i, :], alpha=0.2, c="blue")

idx = (-0.5<=rc.wout) & (rc.wout<=0.5)
print(idx)
for i in np.arange(rc.n+1)[idx]:
        ax2.plot(xs_pred.T[i, :], alpha=0.1, c="green")

plt.legend()
plt.show()