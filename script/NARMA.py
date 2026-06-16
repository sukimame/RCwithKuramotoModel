import numpy as np
import matplotlib.pyplot as plt

def generate_narma10(n_samples, seed=None):

    n=10
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random input uniformly distributed in [0, 0.5]
    u = np.random.uniform(0, 0.5, n_samples + n)
    
    # Initialize output
    y = np.zeros(n_samples + n)
    
    # Generate NARMA-10 sequence
    for t in range(n, n_samples + n):
        # Sum of past 10 outputs
        sum_y = np.sum(y[t-n:t])
        
        # NARMA-10 equation
        y[t] = (0.3 * y[t-1] + 
                0.05 * y[t-1] * sum_y + 
                1.5 * u[t-n] * u[t-1] + 
                0.1)
        
        #y[t] = np.sum(u[t-10:t])/n
                
        #y[t] = u[t-1]

    # Return only the relevant portion (skip first 10 timesteps used for initialization)
    return u[n:], y[n:]

def create_narma10_dataset(train_samples=5000, test_samples=1000, seed=42):
    # Generate training data
    u_train, y_train = generate_narma10(train_samples, seed=seed)
    
    # Generate test data (use different seed to ensure different data)
    u_test, y_test = generate_narma10(test_samples, seed=seed+1 if seed is not None else None)
    
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
    # Example usage
    train_data, test_data = create_narma10_dataset(train_samples=5000, test_samples=1000)
    
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
    
    # Show first few samples
    print("\nFirst 5 samples of training data:")
    print("Input:", train_data['input'][:5])
    print("Target:", train_data['target'][:5])

    plt.plot(test_data['target'])
    plt.show()