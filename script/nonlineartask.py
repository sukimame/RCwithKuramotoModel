from kuramotoRC_Zuo import KURAMOTO_RC
import matplotlib.pyplot as plt
import numpy as np
from NARMA import create_narma10_dataset

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
