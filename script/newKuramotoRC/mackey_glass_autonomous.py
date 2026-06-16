import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from MackeyGlass import create_mackey_glass_dataset
from kuramotoRCEXRK4 import KURAMOTORCEX

dt = 0.01
rc = KURAMOTORCEX(n=100, dt=dt, alpha=0.1, K=6, s=1)

rng = np.random.default_rng()
rc.k = rng.uniform(0, 6, (100, 100))

washout = 200
train   = 100
test    = 50   # 自立予測の長さ
timeSec = washout + train + test

data, _ = create_mackey_glass_dataset(
    train_samples=int(timeSec / dt), test_samples=1,
    horizon=1, tau=17, dt=0.1, seed=42
)

# --- ウォームアップ ---
rc.washout(data['input'][:int(washout / dt)])

# --- 訓練フェーズ (teacher-forced) ---
xs = rc.batch_update(data['input'][int(washout / dt):int((washout + train) / dt)])
ts = data['target'][int(washout / dt):int((washout + train) / dt)]

# 特徴量を正規化 (bias列=0はそのまま)
xs_mean = xs.mean(axis=0)
xs_std  = xs.std(axis=0)
xs_std[xs_std < 1e-8] = 1.0  # 定数列(bias)のゼロ除算防止

xs_norm = xs.copy()
xs_norm[:, 1:] = (xs[:, 1:] - xs_mean[1:]) / xs_std[1:]

# 正規化済み特徴量でリッジ回帰 (λ大きめで安定化)
lambda_reg = 1.0
rc.wout = np.linalg.solve(
    xs_norm.T @ xs_norm + lambda_reg * np.eye(xs_norm.shape[1]),
    xs_norm.T @ ts
)

# --- 自立予測フェーズ (closed-loop) ---
T_pred = int(test / dt)
pred_autonomous = np.zeros(T_pred)

x_t = data['input'][int((washout + train) / dt)]  # 自立予測の起点

for i in range(T_pred):
    rc.updateTheta(x_t)
    x_feat_raw = np.concatenate([[1.0], rc.d_theta * int(1 / rc.dt) - rc.omega])
    x_feat_norm = x_feat_raw.copy()
    x_feat_norm[1:] = (x_feat_raw[1:] - xs_mean[1:]) / xs_std[1:]  # 訓練時と同じ正規化
    y_t = x_feat_norm @ rc.wout
    pred_autonomous[i] = y_t
    x_t = y_t  # 予測値を次の入力へ

target_autonomous = data['target'][int((washout + train) / dt): int((washout + train) / dt) + T_pred]

mse  = np.mean((target_autonomous - pred_autonomous) ** 2)
rmse = np.sqrt(mse)
nrmse = rmse / np.std(target_autonomous)

print(f"Autonomous prediction  NRMSE: {nrmse:.6f}")

t_axis = np.arange(T_pred) * dt

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

ax0 = axes[0]
ax0.plot(t_axis, target_autonomous, label="target (true)", linewidth=1.2)
ax0.plot(t_axis, pred_autonomous,   label="autonomous pred", linestyle="--", linewidth=1.2)
ax0.set_ylabel("x(t)")
ax0.legend(loc="upper right")
ax0.set_title(f"Mackey-Glass autonomous prediction  (NRMSE={nrmse:.4f})")

ax1 = axes[1]
ax1.plot(t_axis, np.abs(target_autonomous - pred_autonomous), color="tomato", linewidth=0.9)
ax1.set_ylabel("|error|")
ax1.set_xlabel("time (s)")
ax1.set_title("Absolute error")

plt.tight_layout()
plt.show()
