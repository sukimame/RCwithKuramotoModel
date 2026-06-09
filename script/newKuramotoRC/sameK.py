"""
6/7
同一接続における、グローバル接続係数lambdaと固有周波数分布の精度への影響を確認

"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from MovingAvg import create_ma_dataset

from kuramotoRCEXRK4 import KURAMOTORCEX

def gridSearch(lambda_, omega, L):
    dt = 0.01
    rc = KURAMOTORCEX(
            n=100,
            dt=dt,
            alpha=0.1,
            K=1,
            s=1,
            lambda_=lambda_
            )
    rc.omega = omega

    washout = 200
    train = 10
    test = 10
    timeSec = washout + train + test 

    data, _ = create_ma_dataset(train_samples=int((timeSec)/dt), test_samples=1, window_size=L, seed=24)

    rc.washout(data['input'][:int(washout/dt)])

    xs = rc.batch_update(data['input'][int(washout/dt):int((washout+train)/dt)])
    rc.ridge(xs, data['target'][int(washout/dt):int((washout+train)/dt)])

    pred = rc.batch_update(data['input'][int((washout+train)/dt):])
    pred = pred @ rc.wout

    mse = np.sum((data['target'][int((washout+train)/dt):] - pred)**2) / pred.shape[0]
    rmse = np.sqrt(mse)

    target_std = np.std(data['target'][int((washout+train)/dt):])
    nrmse = rmse / target_std

    x, y = rc.orderParam(1)
    return nrmse, np.median(rc.wout), np.sqrt(x**2+y**2)


"""
for l in [5, 10, 15, 50, 100]:
    for i, lambda_ in enumerate([0,2,4.5,5.0,5.5,6,6.5,7,8,9]):
        for j, sd in enumerate(np.linspace(0, 1, 10)): 
            omega = rng.normal(0.2, sd, 100)
            nrmse, woutMed, R1 = gridSearch(lambda_, omega, l)

            data[10*i+j] = np.array([lambda_, sd, nrmse, woutMed, R1])
            print(f"{(10*i+(j+1))}% ended")

np.save('/Users/kondolab/myRepos/RCwithKuramotoModel/expData/sameK_June7_', data)
"""

Ls = [1, 2, 5, 10, 20, 50, 100, 200]
lambda_s = [0, 0.5, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 10]
log_sds = np.round(np.logspace(np.log10(0.05), np.log10(1), 8), 4).tolist()
sds = [0] + log_sds # → [0, 0.05, 0.1, 0.2, 0.4, 0.79, 1.58, 3.16, 5.0] (概算)

length = len(Ls)*len(lambda_s)*len(sds)
data = np.zeros((length, 6))
rng = np.random.default_rng()

omegaDict = {}
omegaDict[0] = np.full(100, 0.2)
for j, sd in enumerate(sds[1:]):
    omegaDict[sd] = rng.normal(0.2, sd, 100)

for k, l in enumerate(Ls):
    for i, lambda_ in enumerate(lambda_s):
        for j, sd in enumerate(sds):
            nrmse, woutMed, R1 = gridSearch(lambda_, omegaDict[sd], l)

            idx = (
                len(lambda_s)*len(sds)*k
                + len(sds)*i
                + j
            )
            data[idx] = np.array([l, lambda_, sd, nrmse, woutMed, R1])
            total = len(Ls)*len(lambda_s)*len(sds)
            count = idx + 1

            print(f"{count}/{total}")

np.save('/Users/kondolab/myRepos/RCwithKuramotoModel/expData/sameK_June7_L_lambda_omega_sdlog', data)
