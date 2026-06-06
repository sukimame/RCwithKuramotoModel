import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# NARMAデータのインポート
sys.path.append(str(Path(__file__).resolve().parent.parent))
from NARMA import create_narma10_dataset

# ----------------------------------------
# 1. データの準備
# ----------------------------------------
length = 500  # シミュレーションするステップ数

# 1. まず train_data と test_data の2つの辞書を受け取る
train_data, test_data = create_narma10_dataset(train_samples=length)

# 2. 辞書の中から 'input' と 'target' の配列を取り出す
u_narma = train_data['input']
target_narma = train_data['target']

# （確認用）これできれいな1次元のNumPy配列になります
print(f"u_narma のサイズ: {u_narma.shape}")
# ----------------------------------------
# 2. パラメータ設定
# ----------------------------------------
omega = 0.1
N = 100  # 振動子の数

# 【ポイント】NARMAの1ステップにおける時間を極力短く設定
T_step = 0.1  # 1ステップにかける物理的な時間
h = 0.01      # RK4の刻み幅（T_step=0.1, h=0.01なら、1ステップにつき10回だけ計算が回るため超高速）

# ----------------------------------------
# 3. モデルの定義
# ----------------------------------------
# ※ u_val (NARMAの現在の入力値) を直接受け取るように変更
def f(theta, u_val):
    coupling = 2.5 * np.sum(np.sin(theta[np.newaxis, :] - theta[:, np.newaxis] + 0.001 * u_val), axis=1) / N
    return omega + coupling

# 1回分のRK4計算（時間は入力に依存しなくなったので、yとhとu_valだけでOK）
def step_rk4(y, h_step, u_val):
    k1 = f(y, u_val)
    k2 = f(y + h_step * k1 / 2, u_val)
    k3 = f(y + h_step * k2 / 2, u_val)
    k4 = f(y + h_step * k3, u_val)
    return y + h_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# ----------------------------------------
# 4. シミュレーション（リザバーの駆動）
# ----------------------------------------
np.random.seed(42)
y_current = np.random.uniform(0, 2*np.pi, N)

# 今後の学習に使う「状態ベクトル（位相の変化量）」を保存するリスト
reservoir_states = []
# グラフ描画用に全履歴も保存
y_history = [y_current.copy()]

# NARMAデータのステップ数だけループ
rk4_steps_per_narma = int(T_step / h)  # 今回は 0.1 / 0.02 = 5回

for k in range(len(u_narma)):
    u_k = u_narma[k]            # 現在のステップの入力
    y_prev = y_current.copy()   # 1ステップ前の位相を保持
    
    # T_step の間（5回）、一定の入力 u_k のもとでRK4を回す (ゼロ次ホールド)
    for _ in range(rk4_steps_per_narma):
        y_current = step_rk4(y_current, h, u_k)
        
    # 【特徴量抽出】ステップ前後での位相の変化量 Δθ を計算して保存！
    delta_theta = y_current - y_prev
    reservoir_states.append(delta_theta)
    
    y_history.append(y_current.copy())

# 機械学習（Ridge回帰など）で使いやすいようにNumPy配列化
# shape は (データ数, 100ノード) になります
reservoir_states = np.array(reservoir_states)
y_history = np.array(y_history)

print(f"特徴量行列 (X) のサイズ: {reservoir_states.shape}")

# ----------------------------------------
# 5. グラフ表示 (今まで通り)
# ----------------------------------------
plt.figure(figsize=(10, 6))

t_vals = np.arange(len(y_history)) * T_step  # 横軸を時間スケールに合わせる

# 0 ～ 2π の範囲に収める
y_wrapped = y_history % (2 * np.pi)
y_wrapped_plot = y_wrapped.copy()

# ジャンプ箇所の線を切断
jumps = np.abs(np.diff(y_wrapped_plot, axis=0)) > np.pi
rows, cols = np.where(jumps)
y_wrapped_plot[rows + 1, cols] = np.nan

# プロット
plt.plot(t_vals, y_wrapped_plot, color='blue', alpha=0.3)
plt.xlabel('Time (t)')
plt.ylabel('Phase (theta)')
plt.ylim(0, 2 * np.pi)
plt.yticks([0, np.pi, 2*np.pi], ['0', '$\pi$', '$2\pi$'])
plt.title(f'Coupled Oscillators Driven by NARMA (T_step={T_step})')
plt.grid(True)
plt.show()