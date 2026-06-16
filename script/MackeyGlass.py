import numpy as np
import matplotlib.pyplot as plt

def generate_mackey_glass(n_samples, tau=17, beta=0.2, gamma=0.1, n=10,
                           dt=1.0, warmup=1000, seed=None):
    """
    Mackey-Glass遅延微分方程式をオイラー法で数値積分して時系列を生成する。
      dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

    Parameters
    ----------
    n_samples : int
        返す時系列の長さ
    tau : float
        遅延時間 (tau=17で弱カオス, tau=30で強カオス)
    beta, gamma, n : float
        方程式パラメータ
    dt : float
        積分刻み幅
    warmup : int
        過渡応答を除くためのウォームアップステップ数
    seed : int or None
    """
    if seed is not None:
        np.random.seed(seed)

    tau_steps = int(tau / dt)
    total = warmup + n_samples
    x = np.zeros(total + tau_steps)

    # 初期条件: [0, tau] 区間を 0.9 付近のランダム値で埋める
    x[:tau_steps] = 0.9 + 0.1 * np.random.rand(tau_steps)

    for t in range(tau_steps, total + tau_steps - 1):
        x_delayed = x[t - tau_steps]
        dxdt = beta * x_delayed / (1.0 + x_delayed ** n) - gamma * x[t]
        x[t + 1] = x[t] + dt * dxdt

    return x[tau_steps + warmup:]


def create_mackey_glass_dataset(train_samples=5000, test_samples=1000,
                                 horizon=1, tau=17, dt=1.0, seed=42):
    """
    Mackey-Glass予測タスクのデータセットを作成する。
    input: x(t),  target: x(t + horizon)

    Parameters
    ----------
    horizon : int
        何ステップ先を予測するか
    """
    total = train_samples + test_samples + horizon
    x = generate_mackey_glass(total, tau=tau, dt=dt, seed=seed)

    u = x[:-horizon]
    y = x[horizon:]

    u_train, y_train = u[:train_samples], y[:train_samples]
    u_test,  y_test  = u[train_samples:train_samples + test_samples], \
                       y[train_samples:train_samples + test_samples]

    train_data = {'input': u_train, 'target': y_train}
    test_data  = {'input': u_test,  'target': y_test}

    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = create_mackey_glass_dataset(
        train_samples=5000, test_samples=1000, horizon=1, tau=17, seed=42
    )

    print("Training data:")
    print(f"  Input shape:  {train_data['input'].shape}")
    print(f"  Target shape: {train_data['target'].shape}")
    print(f"  Input range:  [{train_data['input'].min():.3f}, {train_data['input'].max():.3f}]")

    print("\nTest data:")
    print(f"  Input shape:  {test_data['input'].shape}")
    print(f"  Target shape: {test_data['target'].shape}")

    plt.figure(figsize=(10, 4))
    plt.plot(test_data['input'][:300],  label="x(t)")
    plt.plot(test_data['target'][:300], label="x(t+1)", linestyle="--", alpha=0.8)
    plt.title("Mackey-Glass time series (first 300 test samples)")
    plt.xlabel("Time Step")
    plt.ylabel("x")
    plt.legend()
    plt.tight_layout()
    plt.show()
