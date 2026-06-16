import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from NARMA import create_narma10_dataset
from MovingAvg import create_ma_dataset
from MackeyGlass import create_mackey_glass_dataset
from DelayTask import create_delay_dataset

class KURAMOTORCEX:
    def __init__(
            self, 
            n=100, 
            dt=0.01, 
            alpha=0.01, 
            K = 0.7,
            s=1,
            gamma=1,
            lambda_=1,
            epsilon=0.01,
            beta=-np.pi/2+0.1,
        ): 
        self.n = n  
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.beta = beta

        rng = np.random.default_rng()

        self.omega = rng.normal(0.2, 1.0, n)
        #self.omega = np.random.uniform(-0.3, 0.7, n)
        self.theta = rng.uniform(0, 2*np.pi, n)
        self.d_theta = np.zeros(n)

        self.wout = np.ones(n+1)

        self.mask = (np.random.rand(n, n) < s).astype(float)
        self.mask *= (1 - np.eye(n))
        #self.k = np.ones((n, n)) * K/n * self.mask
        #self.k = rng.uniform(0, K, (n, n)) * self.mask
        #self.k = rng.normal(0.6, 0.4, (n, n)) * self.mask

        self.k = rng.uniform(-1, 1, (n, n)) * self.mask

    # --- 新規追加: 任意の位相状態(theta_state)における変化率 dy/dt を計算する関数 ---
    def _calc_dtheta(self, theta_state, u):
        # 差分行列 (N, N) を作成
        diff = theta_state[np.newaxis, :] - theta_state[:, np.newaxis] 
        # d_theta/dt を計算して返す
        return self.omega + self.lambda_ * np.sum(self.k * np.sin(diff + self.alpha * u), axis=1) #+ 0.1*u

    # --- 変更: RK4による状態更新 ---
    def updateTheta(self, u=0):
        # RK4の4ステップ計算
        k1 = self._calc_dtheta(self.theta, u)
        k2 = self._calc_dtheta(self.theta + 0.5 * self.dt * k1, u)
        k3 = self._calc_dtheta(self.theta + 0.5 * self.dt * k2, u)
        k4 = self._calc_dtheta(self.theta + self.dt * k3, u)
        
        # このステップにおける平均的な位相の変化量（速度）
        self.d_theta = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        # 位相の更新 (元のコードの gamma を乗算する仕様を残しています)
        self.theta = self.gamma * self.theta + self.dt * self.d_theta
    
    def updateK(self, u):
        diff = self.theta[np.newaxis, :] - self.theta[:, np.newaxis]  # 位相差
        d_k = -self.epsilon * np.sin(diff + self.beta)
        self.k = self.k + self.dt * d_k

        # クリッピング |k_ij| <= 1
        self.k = np.clip(self.k, -1, 1)
        self.k *= self.mask

    def plot_k_dist(self, bins=50):
        k_vals = self.k[self.mask.astype(bool)]  # マスクで存在するエッジのみ
        _, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].hist(k_vals, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
        axes[0].set_xlabel("k_ij")
        axes[0].set_ylabel("count")
        axes[0].set_title(f"k distribution  (n_edges={len(k_vals)}, mean={k_vals.mean():.3f}, std={k_vals.std():.3f})")

        im = axes[1].imshow(self.k, aspect="auto", cmap="RdBu_r",
                            vmin=-np.abs(self.k).max(), vmax=np.abs(self.k).max())
        plt.colorbar(im, ax=axes[1], label="k_ij")
        axes[1].set_xlabel("j (source)")
        axes[1].set_ylabel("i (target)")
        axes[1].set_title("k matrix")

        plt.tight_layout()
        plt.show()

    def orderParam(self, n=1):
        x = np.mean(np.sin(self.theta * n))
        y = np.mean(np.cos(self.theta * n))
        return x, y

    def washout(self, inputs):
        x, y = self.orderParam(1)
        #print(f"Washout start order parameter: {np.sqrt(x**2+y**2):.4f}")
        for i, u in enumerate(inputs):
            self.updateTheta(u)
            #self.updateK(u=u)

    def batch_update(self, inputs):
        xs = np.zeros((inputs.shape[-1], self.n+1))
        xs[:, 0] = 1

        for i, u in enumerate(inputs):
            self.updateTheta(u)
            # RK4で求めた正確な d_theta を特徴量として使用する
            xs[i, 1:] = self.d_theta * int(1/self.dt) - self.omega
            
        x, y = self.orderParam(1)
        #print(f"Batch end order parameter: {np.sqrt(x**2+y**2):.4f}")
          
        return xs
    
    def ridge(self, xs, ts):
        self.wout = np.linalg.pinv(xs.T @ xs + 1e-4 * np.eye(xs.shape[1])) @ xs.T @ ts


if __name__=="__main__":
    dt = 0.01
    rc = KURAMOTORCEX(
            n=100,
            dt=dt,
            alpha=0.1,
            K=1,
            s=1
            )

    washout = 200
    train = 10
    test = 10
    timeSec = washout + train + test 
    time = np.arange(0, timeSec, dt)

    #data, _ = create_narma10_dataset(train_samples=int((timeSec)/dt), test_samples=1, seed=24)
    #data, _ = create_ma_dataset(train_samples=int((timeSec)/dt), test_samples=1, window_size=10, seed=24)
    data, _ = create_mackey_glass_dataset(train_samples=int((timeSec)/dt), test_samples=1, seed=24)
    #data, _ = create_delay_dataset(int((timeSec)/dt), delay=5, seed=24)

    rc.washout(data['input'][:int(washout/dt)])

    xs = rc.batch_update(data['input'][int(washout/dt):int((washout+train)/dt)])
    rc.ridge(xs, data['target'][int(washout/dt):int((washout+train)/dt)])

    pred = rc.batch_update(data['input'][int((washout+train)/dt):])
    pred = pred @ rc.wout

    mse = np.sum((data['target'][int((washout+train)/dt):] - pred)**2) / pred.shape[0]
    rmse = np.sqrt(mse)

    target_std = np.std(data['target'][int((washout+train)/dt):])
    nrmse = rmse / target_std
    nmse = mse / np.var(data['target'][int((washout+train)/dt):])

    print(np.mean(rc.theta % 2*np.pi), np.std(rc.theta % 2*np.pi))
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"NRMSE: {nrmse:.4f}")
    print(f"NMSE: {nmse:.4f}")

    print("mean, pred and target", np.mean(pred), np.mean(data['target'][int((washout+train)/dt):]))
    print("std, pred and target", np.std(pred), target_std)

    print(np.median(rc.wout), np.std(rc.wout))
    rc.plot_k_dist(bins=50)

    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(pred, label="pred")
    ax1.plot(data['target'][int((washout+train)/dt):], alpha=0.5)

    plt.legend()
    plt.show()


