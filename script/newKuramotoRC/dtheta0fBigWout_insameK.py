import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from NARMA import create_narma10_dataset
from MovingAvg import create_ma_dataset
from DelayTask import create_delay_dataset
from MackeyGlass import create_mackey_glass_dataset

from kuramotoRCEXRK4 import KURAMOTORCEX

dt = 0.01
rc = KURAMOTORCEX(
        n=100,
        dt=dt,
        alpha=0.1,
        K=6,
        s=1
        )

rng = np.random.default_rng()
rc.k = rng.uniform(0, 1, (100, 100))
washout = 200
train = 10
test = 1
timeSec = washout + train + test
time = np.arange(0, timeSec, dt)

#data, _ = create_narma10_dataset(train_samples=int((timeSec)/dt), test_samples=1, seed=24)
#data, _ = create_ma_dataset(train_samples=int((timeSec)/dt), test_samples=1, window_size=10, seed=24)
data, _ = create_mackey_glass_dataset(train_samples=int(timeSec/dt), test_samples=1,
                                      horizon=1, tau=17, dt=1.0, seed=42)

rc.washout(data['input'][:int(washout/dt)])

xs = rc.batch_update(data['input'][int(washout/dt):int((washout+train)/dt)])
rc.ridge(xs, data['target'][int(washout/dt):int((washout+train)/dt)])

# 予測フェーズ: 位相と位相変化量の履歴を収集するため手動ループ
pred_input = data['input'][int((washout+train)/dt):]
T_pred = len(pred_input)

xs_pred = np.zeros((T_pred, rc.n + 1))
xs_pred[:, 0] = 1
theta_history = np.zeros((T_pred, rc.n))
d_theta_history = np.zeros((T_pred, rc.n))

for i, u in enumerate(pred_input):
    rc.updateTheta(u)
    xs_pred[i, 1:] = rc.d_theta * int(1 / rc.dt) - rc.omega
    theta_history[i] = rc.theta.copy()
    d_theta_history[i] = rc.d_theta.copy()

pred = xs_pred @ rc.wout

mse = np.sum((data['target'][int((washout+train)/dt):] - pred)**2) / pred.shape[0]
rmse = np.sqrt(mse)
target_std = np.std(data['target'][int((washout+train)/dt):])
nrmse = rmse / target_std

print(np.mean(rc.theta % 2*np.pi), np.std(rc.theta % 2*np.pi))
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"NRMSE: {nrmse:.6f}")
print("mean, pred and target", np.mean(pred), np.mean(data['target'][int((washout+train)/dt):]))
print("std, pred and target", np.std(pred), target_std)
print(np.median(rc.wout), np.std(rc.wout), rc.wout)

# wout (bias項を除く) による色分け
wout_vals = rc.wout[1:]
t_axis = np.arange(T_pred) * dt

# 円環平均位相からの位相差 ([-π, π] に折り返す)
circular_mean = np.angle(np.mean(np.exp(1j * theta_history), axis=1, keepdims=True))
theta_wrapped = np.angle(np.exp(1j * theta_history))
phase_diff = np.angle(np.exp(1j * (theta_wrapped - circular_mean)))

# 秩序パラメータ R (0: 非同期, 1: 完全同期)
order_param = np.abs(np.mean(np.exp(1j * theta_history), axis=1))

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# --- 上段: 予測と目標 ---
ax0 = axes[0]
ax0.plot(t_axis, data['target'][int((washout+train)/dt):], label="target")
ax0.plot(t_axis, data['input'][int((washout+train)/dt):], label="input", alpha=0.5)
ax0.plot(t_axis, pred, label="pred", linestyle="--")
ax0.set_ylabel("output")
ax0.legend(loc="upper right")
ax0.set_title(f"Prediction vs Target  (NRMSE={nrmse:.4f})")

# --- 中段: 位相差 θ_i − θ̄ ---
ax1 = axes[1]
for i in range(rc.n):
    if wout_vals[i] > 0.5:
        ax1.plot(t_axis, phase_diff[:, i], alpha=0.3, c="red", linewidth=0.8)
    elif wout_vals[i] < -0.5:
        ax1.plot(t_axis, phase_diff[:, i], alpha=0.3, c="blue", linewidth=0.8)
    else:
        ax1.plot(t_axis, phase_diff[:, i], alpha=0.01, c="green", linewidth=0.8)

ax1.set_ylabel(r"phase diff $\theta_i - \bar{\theta}$")
ax1_r = ax1.twinx()
ax1_r.plot(t_axis, order_param, c="black", linewidth=1.5, label="R (order param)")
ax1_r.set_ylim(0, 1.1)
ax1_r.set_ylabel("R")
ax1_r.legend(loc="upper right")
ax1.set_title("Phase Difference + Order Parameter R — red: wout>0.5, blue: wout<-0.5")

# --- 下段: 位相の変化量 dθ/dt ---
ax2 = axes[2]
for i in range(rc.n):
    if wout_vals[i] > 0.5:
        ax2.plot(t_axis, d_theta_history[:, i], alpha=0.3, c="red", linewidth=0.8)
    elif wout_vals[i] < -0.5:
        ax2.plot(t_axis, d_theta_history[:, i], alpha=0.3, c="blue", linewidth=0.8)
    else:
        ax2.plot(t_axis, d_theta_history[:, i], alpha=0.1, c="green", linewidth=0.8)
ax2.set_ylabel(r"phase velocity $d\theta/dt$")
ax2.set_xlabel("time (s)")
ax2.set_title("Phase Velocity (dθ/dt) — red: wout>0.5, blue: wout<-0.5")

plt.tight_layout()
plt.show()
