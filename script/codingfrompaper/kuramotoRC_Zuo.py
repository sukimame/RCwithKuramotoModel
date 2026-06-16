import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from NARMA import create_narma10_dataset
from MackeyGlass import create_mackey_glass_dataset

def adjust_spectral_radius(K, target_radius):
    """スペクトル半径をtarget_radiusに正規化（論文Algorithm 1の6行目）"""
    eigenvalues = np.linalg.eigvals(K)
    current_radius = np.max(np.abs(eigenvalues))
    if current_radius < 1e-10:
        return K
    return K * (target_radius / current_radius)


class KURAMOTO_RC_ZUO:
    def __init__(
            self,
            n,
            dt=1.0,
            alpha=1.0,       # 入力結合強度（論文ではWinを省略しているためalphaで代用）
            lambda_=4.0,     # 論文Table I: NARMA10ではλ=4.0
            radius=1.1,      # 論文のスペクトル半径ρ
            epsilon=0.1,     # 論文: ε=0.1
            beta=np.pi/2,    # 論文: β=π/2をデフォルト（NARMA10で使用）
            density=0.05,    # 論文: 接続密度s=0.05（5%）
        ):

        self.dt = dt
        self.n = n
        self.alpha = alpha
        self.lambda_ = lambda_
        self.radius = radius
        self.epsilon = epsilon
        self.beta = beta

        rng = np.random.default_rng()

        # 論文: Ω ~ 標準正規分布 N(0,1)
        self.omega = rng.normal(0, 1, n)

        # 論文: Θ = 0（全ノード位相ゼロ初期化）
        self.theta = np.zeros(n)

        self.wout = np.ones(2 * n + 1)

        # 論文: 密度s=0.05のスパース接続、非ゼロ要素はUniform[-1,1]
        self.mask = (np.random.rand(n, n) < density).astype(float)
        self.mask *= (1 - np.eye(n))  # 自己結合なし
        self.k = rng.uniform(-1, 1, (n, n)) * self.mask

        # 論文: 初期化後にスペクトル半径をρに正規化
        self.k = adjust_spectral_radius(self.k, self.radius)
        self.k *= self.mask  # マスクを再適用（正規化でゼロ要素が変わる場合に備え）

    def _phase_diff(self):
        """theta_j - theta_i の行列（論文式(1)(2)のΔθ）"""
        return self.theta[np.newaxis, :] - self.theta[:, np.newaxis]

    def updateTheta(self, u=0):
        """論文式(1): 位相更新（入力uはsin内に加算）"""
        diff = self._phase_diff()
        d_theta = self.omega + self.lambda_ * np.sum(
            self.k * np.sin(diff + self.alpha * u), axis=1
        )
        self.theta = self.theta + self.dt * d_theta

    def updateK(self):
        """論文式(2): 接続重み更新 + クリッピング + スペクトル半径正規化"""
        diff = self._phase_diff()
        d_k = -self.epsilon * np.sin(diff + self.beta)
        self.k = self.k + self.dt * d_k

        # クリッピング |k_ij| <= 1
        self.k = np.clip(self.k, -1, 1)
        self.k *= self.mask

        # 論文Algorithm 1の6行目: スペクトル半径を再正規化
        #self.k = adjust_spectral_radius(self.k, self.radius)
        #self.k *= self.mask

    def orderParam(self):
        """1次のKuramotoオーダーパラメータ r"""
        z = np.mean(np.exp(1j * self.theta))
        return np.abs(z)

    def washout(self, inputs):
        """
        論文のautonomous development stage:
        washout期間中にupdateKを実行してリザバー構造を形成する
        """
        op = []
        for u in inputs:
            self.updateTheta(u)
            self.updateK()          # ← 論文の核心：washout中に接続を更新
            op.append(self.orderParam())
        return op

    def batch_update(self, inputs):
        """訓練・テスト期間：接続固定でノード状態のみ更新"""
        xs = np.zeros((len(inputs), 2 * self.n + 1))
        xs[:, 0] = 1  # バイアス項

        op = []
        for i, u in enumerate(inputs):
            self.updateTheta(u)
            xs[i, 1:self.n + 1] = np.sin(self.theta)
            xs[i, self.n + 1:]  = np.cos(self.theta)
            op.append(self.orderParam())

        return xs, op

    def ridge(self, xs, ts, reg=1e-3):
        """リッジ回帰で出力重みを学習"""
        self.wout = np.linalg.solve(
            xs.T @ xs + reg * np.eye(xs.shape[1]),
            xs.T @ ts
        )


if __name__ == "__main__":
    dt = 1

    # 論文Table I: NARMA10の設定
    rc = KURAMOTO_RC_ZUO(
        n=100,
        dt=dt,
        alpha=1.0,
        lambda_=4.0,
        radius=1.1,
        epsilon=0.1,
        beta=-np.pi / 2,
        density=0.05,
    )

    # 論文Table I: L_adev=100, L_train=900, L_test=500
    washout = 1000
    train   = 2900
    test    = 500
    total   = washout + train + test

    #data, _ = create_narma10_dataset(
    #    train_samples=total, test_samples=1, seed=42
    #)
    data, _ = create_mackey_glass_dataset(
        train_samples=total, test_samples=1, seed=42
    )
    u_all = data['input']
    y_all = data['target']

    # Autonomous development stage（washout）
    op1 = rc.washout(u_all[:washout])
    print(f"[After washout] order param: {op1[-1]:.4f}")
    print(f"[After washout] K mean: {np.mean(rc.k):.4f}, std: {np.std(rc.k):.4f}")

    # 訓練
    xs, op2 = rc.batch_update(u_all[washout:washout + train])
    rc.ridge(xs, y_all[washout:washout + train])

    # テスト
    xs_test, op3 = rc.batch_update(u_all[washout + train:])
    pred = xs_test @ rc.wout

    # 評価
    target = y_all[washout + train:]
    mse   = np.mean((target - pred) ** 2)
    rmse  = np.sqrt(mse)
    nrmse = rmse / np.std(target)

    print(f"\nMSE:   {mse:.6f}")
    print(f"RMSE:  {rmse:.6f}")
    print(f"NRMSE: {nrmse:.6f}")

    # プロット
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    axes[0].plot(target, label='target', alpha=0.7)
    axes[0].plot(pred, label='pred', alpha=0.7, linestyle='--')
    axes[0].set_title(f'Prediction vs Target  (NRMSE={nrmse:.4f})')
    axes[0].legend()

    axes[1].plot(op1, label='washout (dev)')
    axes[1].plot(range(washout, washout + train), op2, label='train')
    axes[1].plot(range(washout + train, total), op3, label='test')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Order Parameter R')
    axes[1].legend()

    axes[2].hist(rc.k[rc.mask > 0], bins=50)
    axes[2].set_title('Distribution of K after development')
    axes[2].set_xlabel('k_ij')

    nz = np.sum(rc.mask > 0)
    sr = np.max(np.abs(np.linalg.eigvals(rc.k)))
    print(f"非ゼロ要素数: {nz}, スペクトル半径: {sr:.4f}")

    plt.tight_layout()
    plt.show()