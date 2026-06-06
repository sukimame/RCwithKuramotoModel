import numpy as np
import matplotlib.pyplot as plt

omega = 0.1
N = 100  # 振動子の数

def u(t):
    return 0.5 * np.sin(0.5 * t)  # 外部入力（例として正弦波）
# 微分方程式 dy/dt = f(t, theta)
def f(t, theta):
    # theta[np.newaxis, :] - theta[:, np.newaxis] で (N, N) の差分行列を作成
    # 結合定数 K=2 を掛け、Nで割る。外力は位相差のシフトとして作用。
    coupling = 2.5 * np.sum(np.sin(theta[np.newaxis, :] - theta[:, np.newaxis] + 0.001 * u(t)), axis=1) / N
    return omega + coupling

# ルンゲ＝クッタ4次法
def runge_kutta_4(f, y0, t0, t_end, h):
    t_values = [t0]
    y_values = [y0.copy()]  # 初期値のコピーを保存
    
    t = t0
    y = y0.copy()  # 元の y0 を書き換えないようにコピー
    
    while t < t_end:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h/2, y + h*k2/2)
        k4 = f(t + h, y + h*k3)
        
        # 新しい配列を生成して更新
        y = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h

        t_values.append(t)
        y_values.append(y.copy())
    
    return np.array(t_values), np.array(y_values)

# パラメータ設定
t0 = 0
t_end = 5

# 初期位相の設定（ランダム分布）
np.random.seed(42)
y0 = np.random.uniform(0, 2*np.pi, N) 
h = 0.05

# 実行
t_vals, y_vals = runge_kutta_4(f, y0, t0, t_end, h)



# --- これより上のシミュレーション部分（runge_kutta_4の実行まで）はそのまま ---

# グラフ表示
plt.figure(figsize=(10, 6))

# 1. まず 0 ～ 2π の範囲に収めた配列を作る
y_wrapped = y_vals % (2 * np.pi)

# 2. 描画用の配列をコピーして用意
y_wrapped_plot = y_wrapped.copy()

# 3. 縦方向（時間方向）の差分を計算し、π以上変化した（＝2πから0へジャンプした）場所を探す
jumps = np.abs(np.diff(y_wrapped_plot, axis=0)) > np.pi

# 4. ジャンプが発生した「次のインデックス」を np.nan にして線を切る
rows, cols = np.where(jumps)
y_wrapped_plot[rows + 1, cols] = np.nan

# 5. プロット（np.nan の部分で線が途切れるため、綺麗な折り返しになる）
plt.plot(t_vals, y_wrapped_plot, color='blue', alpha=0.3)

plt.xlabel('Time (t)')
plt.ylabel('Phase (theta)')
plt.ylim(0, 2 * np.pi)
plt.yticks([0, np.pi, 2*np.pi], ['0', '$\pi$', '$2\pi$']) # 目盛りを分かりやすくするおまけ
plt.title('Coupled Oscillators (Kuramoto Model)')
plt.grid(True)
plt.show()

