import numpy as np
import matplotlib.pyplot as plt

def generate_delay_task(n_samples, delay=1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    u = np.random.uniform(0, 0.5, n_samples + delay)
    y = u[:n_samples]  # y[t] = u[t - delay]

    return u[delay:], y

def create_delay_dataset(train_samples=5000, test_samples=1000, delay=1, seed=42):
    u_train, y_train = generate_delay_task(train_samples, delay, seed=seed)
    u_test, y_test = generate_delay_task(test_samples, delay, seed=seed+1 if seed is not None else None)

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
    train_data, test_data = create_delay_dataset(train_samples=5000, test_samples=1000, delay=3)

    print("Training data:")
    print(f"  Input shape: {train_data['input'].shape}")
    print(f"  Target shape: {train_data['target'].shape}")
    print(f"  Input range: [{train_data['input'].min():.3f}, {train_data['input'].max():.3f}]")
    print(f"  Target range: [{train_data['target'].min():.3f}, {train_data['target'].max():.3f}]")

    print("\nTest data:")
    print(f"  Input shape: {test_data['input'].shape}")
    print(f"  Target shape: {test_data['target'].shape}")

    print("\nFirst 5 samples of training data:")
    print("Input: ", train_data['input'][:5])
    print("Target:", train_data['target'][:5])

    plt.figure(figsize=(10, 4))
    plt.plot(test_data['input'][:100], label="input u(t)")
    plt.plot(test_data['target'][:100], label=f"target u(t-{3})", linestyle="--")
    plt.title("Delay Task (First 100 samples of Test Data)")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()
