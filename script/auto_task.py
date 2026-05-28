from script.kuramotoRC_chiba import KURAMOTO_RC, almost_periodic
import matplotlib.pyplot as plt
import numpy as np

n=500
rc = KURAMOTO_RC(n=n)

dt = 0.01

washout = 240
train = 32
test = 32
timeSec = washout + train + test # 240s for training, 32s for training, 32s for testing
time = np.arange(0, timeSec, dt)

inputs = almost_periodic(time)
rc.batch_update(inputs[:int(washout/dt)])

ts = almost_periodic(time - 100)

xs = np.zeros((int(train/dt), n*2+1))
for i in range(1, int((train)/dt)):
    u = inputs[int(washout/dt)+i]
    xs[i] = rc.autonomous_update(u)

rc.ridge(xs, ts[int((washout)/dt):int((washout+train)/dt)])

pred = np.zeros(int(test/dt))
for i in range(int(test/dt)):
    u = pred[i-1] if i > 0 else inputs[int((washout+train)/dt)]
    x = rc.autonomous_update(u)
    pred[i] = x @ rc.wout

acc = np.sum((ts[int((washout+train)/dt):] - pred)**2) / pred.shape[0]

print("MSE", acc)

plt.plot(pred)
plt.plot(ts[int((washout+train)/dt):int((washout+train+test)/dt)+pred.shape[0]])
plt.show()
