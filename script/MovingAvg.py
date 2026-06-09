import numpy as np
import matplotlib.pyplot as plt

def generate_moving_average(n_samples, window_size=10, seed=None):
    """
    指定されたサンプル数とウィンドウサイズで移動平均データセットを生成する。
    """
    if seed is not None:
        np.random.seed(seed)
    
    # [0, 0.5]の範囲でランダムな入力を生成（初期化用の期間分も多めに生成）
    u = np.random.uniform(0, 0.5, n_samples + window_size)
    
    # 出力を初期化
    y = np.zeros(n_samples + window_size)
    
    # 移動平均を計算
    for t in range(window_size, n_samples + window_size):
        # 過去 window_size 分の入力の平均をとる
        y[t] = np.sum(u[t-window_size:t]) / window_size

    # 最初の初期化用ステップをスキップして、必要な部分だけを返す
    return u[window_size:], y[window_size:]

def create_ma_dataset(train_samples=5000, test_samples=1000, window_size=10, seed=42):
    """
    移動平均の訓練データとテストデータを作成する。
    """
    # 訓練データの生成
    u_train, y_train = generate_moving_average(train_samples, window_size, seed=seed)
    
    # テストデータの生成（異なるデータになるようにシードをずらす）
    u_test, y_test = generate_moving_average(test_samples, window_size, seed=seed+1 if seed is not None else None)
    
    train_data = {
        'input': u_train,
        'target': y_train
    }
    
    test_data = {
        'input': u_test,
        'target': y_test
    }
    
    return train_data, test_data


if __name__ == "__main__":
    # 使用例: ウィンドウサイズ10の移動平均データセットを作成
    train_data, test_data = create_ma_dataset(train_samples=5000, test_samples=1000, window_size=10)
    
    print("Training data:")
    print(f"  Input shape: {train_data['input'].shape}")
    print(f"  Target shape: {train_data['target'].shape}")
    print(f"  Input range: [{train_data['input'].min():.3f}, {train_data['input'].max():.3f}]")
    print(f"  Target range: [{train_data['target'].min():.3f}, {train_data['target'].max():.3f}]")
    
    print("\nTest data:")
    print(f"  Input shape: {test_data['input'].shape}")
    print(f"  Target shape: {test_data['target'].shape}")
    print(f"  Input range: [{test_data['input'].min():.3f}, {test_data['input'].max():.3f}]")
    print(f"  Target range: [{test_data['target'].min():.3f}, {test_data['target'].max():.3f}]")
    
    # 最初の5サンプルの確認
    print("\nFirst 5 samples of training data:")
    print("Input:", train_data['input'][:5])
    print("Target:", train_data['target'][:5])

    # テストデータのターゲット（移動平均）をプロット
    plt.figure(figsize=(10, 4))
    plt.plot(test_data['target'][:200], label='Moving Average Target')
    plt.title("Moving Average Series (First 200 samples of Test Data)")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()