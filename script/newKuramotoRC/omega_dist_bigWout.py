import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from MovingAvg import create_ma_dataset
from DelayTask import create_delay_dataset
from kuramotoRCEXRK4 import KURAMOTORCEX

dt = 0.01
rc = KURAMOTORCEX(n=100, dt=dt, alpha=0.1, K=6, s=1)

rng = np.random.default_rng()
rc.k = rng.uniform(0, 6, (100, 100))

washout = 150
train = 5
test = 1
timeSec = washout + train + test

data, _ = create_delay_dataset(train_samples=int(timeSec / dt), test_samples=1, delay=5, seed=24)

rc.washout(data['input'][:int(washout / dt)])
xs = rc.batch_update(data['input'][int(washout / dt):int((washout + train) / dt)])
rc.ridge(xs, data['target'][int(washout / dt):int((washout + train) / dt)])

# 予測フェーズで位相履歴を収集
pred_input = data['input'][int((washout + train) / dt):]
T_pred = len(pred_input)
theta_history = np.zeros((T_pred, rc.n))
for i, u in enumerate(pred_input):
    rc.updateTheta(u)
    theta_history[i] = rc.theta.copy()

# 円環平均からの位相差の標準偏差 (ノードごとに1値)
circular_mean = np.angle(np.mean(np.exp(1j * theta_history), axis=1, keepdims=True))
theta_wrapped = np.angle(np.exp(1j * theta_history))
phase_diff = np.angle(np.exp(1j * (theta_wrapped - circular_mean)))  # (T, n)
phase_diff_std = phase_diff.std(axis=0)  # shape (n,)

wout_vals = rc.wout[1:]  # バイアス除く, shape (100,)
omega = rc.omega          # 固有周波数, shape (100,)

thresh = 0.5
mask_pos = wout_vals >  thresh
mask_neg = wout_vals < -thresh
mask_mid = ~mask_pos & ~mask_neg

print(f"wout >  {thresh}: {mask_pos.sum()} nodes")
print(f"wout < -{thresh}: {mask_neg.sum()} nodes")
print(f"others:           {mask_mid.sum()} nodes")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- 左: ヒストグラム ---
ax = axes[0]
bins = np.linspace(omega.min() - 0.2, omega.max() + 0.2, 30)
ax.hist(omega[mask_mid], bins=bins, color="green",  alpha=0.5, label=f"|wout|≤{thresh}")
ax.hist(omega[mask_pos], bins=bins, color="red",    alpha=0.7, label=f"wout> {thresh}")
ax.hist(omega[mask_neg], bins=bins, color="blue",   alpha=0.7, label=f"wout<-{thresh}")
ax.set_xlabel("omega_i (natural frequency)")
ax.set_ylabel("count")
ax.set_title("Distribution of omega_i by wout group")
ax.legend()

# --- 右: wout vs omega の散布図 ---
ax2 = axes[1]
ax2.scatter(omega[mask_mid], wout_vals[mask_mid], c="green", alpha=0.5, s=20, label=f"|wout|≤{thresh}")
ax2.scatter(omega[mask_pos], wout_vals[mask_pos], c="red",   alpha=0.8, s=30, label=f"wout> {thresh}")
ax2.scatter(omega[mask_neg], wout_vals[mask_neg], c="blue",  alpha=0.8, s=30, label=f"wout<-{thresh}")
ax2.axhline( thresh, color="red",  linestyle="--", linewidth=0.8)
ax2.axhline(-thresh, color="blue", linestyle="--", linewidth=0.8)
ax2.set_xlabel("omega_i (natural frequency)")
ax2.set_ylabel("wout_i")
ax2.set_title("wout vs omega_i")
ax2.legend()

# --- 右: wout vs 位相差の標準偏差 ---
ax3 = axes[2]
ax3.scatter(phase_diff_std[mask_mid], wout_vals[mask_mid], c="green", alpha=0.5, s=20, label=f"|wout|≤{thresh}")
ax3.scatter(phase_diff_std[mask_pos], wout_vals[mask_pos], c="red",   alpha=0.8, s=30, label=f"wout> {thresh}")
ax3.scatter(phase_diff_std[mask_neg], wout_vals[mask_neg], c="blue",  alpha=0.8, s=30, label=f"wout<-{thresh}")
ax3.axhline( thresh, color="red",  linestyle="--", linewidth=0.8)
ax3.axhline(-thresh, color="blue", linestyle="--", linewidth=0.8)
ax3.set_xlabel("std of phase diff (theta_i - theta_mean)")
ax3.set_ylabel("wout_i")
ax3.set_title("wout vs std(phase diff)")
ax3.legend()

plt.tight_layout()
plt.show()
