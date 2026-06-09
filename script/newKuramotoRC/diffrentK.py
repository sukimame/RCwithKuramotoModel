"""
6/9
ランダム接続における短期記憶能力を調査
"""
import numpy as np
from kuramotoRCEXRK4 import KURAMOTORCEX

def RandomInput(n, delay, seed=24):
    np.random.seed(seed)
    raw = np.random.uniform(0, 0.5, n)
    return raw[delay:], raw[:-delay]

def gridSearch(k_scale, omega, delay):  # lambda_ → k_scale に変更
    dt = 0.01
    rc = KURAMOTORCEX(
        n=100,
        dt=dt,
        alpha=0.01,
        K=1,
        s=1,
        lambda_=1  # lambda_ は固定（不要なら引数から削除可）
    )
    rc.omega = omega
    rc.k = rng.uniform(0, k_scale, (100, 100)) * rc.mask  # 0〜k_scale に変更

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

delays = list(range(1, 21))  # 1〜20ステップ

k_scales = np.round(np.linspace(0, 1, 11), 4).tolist()   # 0を含む0〜1の11段階
log_sds = np.round(np.logspace(np.log10(0.05), np.log10(1), 7), 4).tolist()
sds = [0] + log_sds  # 8値に増やす

length = len(delays) * len(k_scales) * len(sds)  # Ls バグも修正
data = np.zeros((length, 6))
rng = np.random.default_rng()

omegaDict = {}
omegaDict[0] = np.full(100, 0.2)
for j, sd in enumerate(sds[1:]):
    omegaDict[sd] = rng.normal(0.2, sd, 100)

for k, delay in enumerate(delays):
    for i, k_scale in enumerate(k_scales):  # lambda_ → k_scale
        for j, sd in enumerate(sds):
            nrmse, woutMed, R1 = gridSearch(k_scale, omegaDict[sd], delay)
            idx = (
                len(k_scales) * len(sds) * k
                + len(sds) * i
                + j
            )
            data[idx] = np.array([delay, k_scale, sd, nrmse, woutMed, R1])
            total = len(delays) * len(k_scales) * len(sds)
            count = idx + 1
            print(f"{count}/{total}")

np.save('/Users/kondolab/myRepos/RCwithKuramotoModel/expData/differentK_June9_delay_omega_sdlog', data)