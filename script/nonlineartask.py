from kuramotoRC import KURAMOTO_RC
import matplotlib.pyplot as plt
import numpy as np

def lowhighSin(t):
    return np.sin(t) + np.sin(10*np.sqrt(2)*t)

rc = KURAMOTO_RC()

dt = 0.01

washout = 240
train = 32
test = 32
timeSec = washout + train + test # 240s for training, 32s for training, 32s for testing
time = np.arange(0, timeSec, dt)

inputs = lowhighSin(time)
rc.batch_update(inputs[:int(washout/dt)])

ts = lowhighSin(time + 100)
xs = rc.batch_update(inputs[int(washout/dt):int((washout+train)/dt)])
rc.ridge(xs, ts[int((washout)/dt):int((washout+train)/dt)])

pred = rc.batch_update(inputs[int((washout+train)/dt):]) @ rc.wout

#pred = np.zeros(test)

acc = np.sum((ts[int((washout+train)/dt):] - pred)**2) / pred.shape[0]

print("MSE", acc)

plt.plot(pred)
plt.plot(ts[int((washout+train)/dt):int((washout+train+test)/dt)+pred.shape[0]])
plt.show()
