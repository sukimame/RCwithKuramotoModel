"""
6/7
同一接続における、グローバル接続係数lambdaと固有周波数分布の精度への影響を確認
・sdは対数スケール
・lambda転移点付近を細かく
・5トライアル（omega・初期位相固定）の中央値を使用
"""
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from MovingAvg import create_ma_dataset
from kuramotoRCEXRK4 import KURAMOTORCEX

N_TRIALS = 5

def gridSearch(lambda_, omega, phase, L):
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
    rc.phase = phase  # 初期位相を固定

    washout = 200
    train   = 10
    test    = 10
    timeSec = washout + train + test

    data, _ = create_ma_dataset(
        train_samples=int((timeSec) / dt),
        test_samples=1,
        window_size=L,
        seed=24,
    )

    rc.washout(data['input'][:int(washout / dt)])
    xs = rc.batch_update(data['input'][int(washout / dt):int((washout + train) / dt)])
    rc.ridge(xs, data['target'][int(washout / dt):int((washout + train) / dt)])

    pred = rc.batch_update(data['input'][int((washout + train) / dt):])
    pred = pred @ rc.wout

    mse        = np.sum((data['target'][int((washout + train) / dt):] - pred) ** 2) / pred.shape[0]
    rmse       = np.sqrt(mse)
    target_std = np.std(data['target'][int((washout + train) / dt):])
    nrmse      = rmse / target_std

    x, y = rc.orderParam(1)
    return nrmse, np.median(rc.wout), np.sqrt(x ** 2 + y ** 2)


# ── パラメータグリッド ────────────────────────────────────────────
Ls = [1, 2, 5, 10, 20, 50, 100, 200]
lambda_s = [0, 0.5, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 10]
log_sds = np.round(np.logspace(np.log10(0.05), np.log10(1), 8), 2).tolist()
sds = [0] + log_sds
# → [0, 0.05, 0.08, 0.13, 0.2, 0.32, 0.5, 0.79, 1.0]

# ── トライアルごとの omega・初期位相を事前に固定生成 ──────────────
rng = np.random.default_rng(seed=42)

omegaDict = {}  # omegaDict[sd][trial]
phaseDict = {}  # phaseDict[trial]

omegaDict[0] = [np.full(100, 0.2) for _ in range(N_TRIALS)]
for sd in sds[1:]:
    omegaDict[sd] = [rng.normal(0.2, sd, 100) for _ in range(N_TRIALS)]

for t in range(N_TRIALS):
    phaseDict[t] = rng.uniform(0, 2 * np.pi, 100)

# ── グリッドサーチ ────────────────────────────────────────────────
length = len(Ls) * len(lambda_s) * len(sds)
# 各条件につき [L, lambda, sd, nrmse_median, woutMed_median, R1_median] を保存
data  = np.zeros((length, 6))
total = length
count = 0

for k, l in enumerate(Ls):
    for i, lambda_ in enumerate(lambda_s):
        for j, sd in enumerate(sds):

            nrmses    = []
            woutMeds  = []
            R1s       = []

            for t in range(N_TRIALS):
                nrmse, woutMed, R1 = gridSearch(
                    lambda_,
                    omegaDict[sd][t],
                    phaseDict[t],
                    l,
                )
                nrmses.append(nrmse)
                woutMeds.append(woutMed)
                R1s.append(R1)
                print(t)

            idx = (
                len(lambda_s) * len(sds) * k
                + len(sds) * i
                + j
            )
            data[idx] = np.array([
                l,
                lambda_,
                sd,
                np.median(nrmses),
                np.median(woutMeds),
                np.median(R1s),
            ])

            count += 1
            print(f"{count}/{total}  L={l}  λ={lambda_}  sd={sd:.4f}  "
                  f"NRMSE_med={np.median(nrmses):.4f}")

np.save(
    '/Users/kondolab/myRepos/RCwithKuramotoModel/expData/sameK_June7_L_lambda_omega_sdlog_trial5',
    data,
)
print("Done.")