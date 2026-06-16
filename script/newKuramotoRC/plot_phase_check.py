import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from NARMA import create_narma10_dataset
from MovingAvg import create_ma_dataset

from kuramotoRCEXRK4 import KURAMOTORCEX

dt = 0.01
rc = KURAMOTORCEX(n=100, dt=dt, alpha=0.1, K=3, s=1)

washout = 200
train = 5
test = 5  # 位相を観察するため長めに
timeSec = washout + train + test
data, _ = create_ma_dataset(train_samples=int(timeSec / dt), test_samples=1, window_size=10, seed=24)

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

t_axis = np.arange(T_pred) * dt

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# --- 上段: 生の位相 θ_i(t) (折り返しなし) ---
ax0 = axes[0]
for i in range(rc.n):
    ax0.plot(t_axis, theta_history[:, i], alpha=0.15, linewidth=0.7, c="steelblue")
ax0.set_ylabel("theta_i (raw)")
ax0.set_title("Raw phase θ_i(t)  — not wrapped")

# --- 中段: 2π に巻き戻した位相 θ_i mod 2π ---
ax1 = axes[1]
theta_wrapped = theta_history % (2 * np.pi)
for i in range(rc.n):
    ax1.plot(t_axis, theta_wrapped[:, i], alpha=0.15, linewidth=0.7, c="darkorange")
ax1.set_ylabel("theta_i mod 2pi")
ax1.set_ylim(0, 2 * np.pi)
ax1.set_yticks([0, np.pi, 2 * np.pi])
ax1.set_yticklabels(["0", "π", "2π"])
ax1.set_title("Wrapped phase θ_i mod 2π")

# --- 下段: ヒートマップ (振動子×時刻, 色=位相 mod 2π) ---
ax2 = axes[2]
im = ax2.imshow(
    theta_wrapped.T,
    aspect="auto",
    origin="lower",
    extent=[t_axis[0], t_axis[-1], 0, rc.n],
    cmap="hsv",
    vmin=0,
    vmax=2 * np.pi,
)
plt.colorbar(im, ax=ax2, label="theta_i")
ax2.set_ylabel("oscillator index")
ax2.set_xlabel("time (s)")
ax2.set_title("Phase heatmap (oscillator × time)")

plt.tight_layout()
plt.show()
