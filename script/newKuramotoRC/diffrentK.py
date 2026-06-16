"""
6/9
ランダム接続における短期記憶能力を調査
"""
import numpy as np
from kuramotoRCEXRK4 import KURAMOTORCEX

def RandomInput(n, delay, seed=24):
    np.random.seed(seed)
    raw = np.random.uniform(0, 0.5, n)
    return raw[delay:], raw[:-delay]  # 修正：正しい向き

def gridSearch(k_scale, delay):
    dt = 0.01
    rc = KURAMOTORCEX(
        n=100,
        dt=dt,
        alpha=0.01,
        K=1,
        s=1,
        lambda_=1
    )
    rc.omega = np.full(100, 0.2)  # sd不要なので均一固定
    rc.k = rng.uniform(0, k_scale, (100, 100)) * rc.mask

    washout = 200
    train = 10
    test = 10
    timeSec = washout + train + test
    data = {}
    data['input'], data['target'] = RandomInput(int(timeSec/dt), delay, seed=24)

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

delays = list(range(1, 21))
k_scales = np.round(np.linspace(0, 20, 161), 4).tolist()  # 0〜10を21段階

length = len(delays) * len(k_scales)
data = np.zeros((length, 5))
rng = np.random.default_rng()

for k, delay in enumerate(delays):
    for i, k_scale in enumerate(k_scales):
        nrmse, woutMed, R1 = gridSearch(k_scale, delay)
        idx = len(k_scales) * k + i
        data[idx] = np.array([delay, k_scale, nrmse, woutMed, R1])
        total = len(delays) * len(k_scales)
        count = idx + 1
        print(f"{count}/{total}")

np.save('/Users/kondolab/myRepos/RCwithKuramotoModel/expData/differentK_June10_kscale20_delay', data)