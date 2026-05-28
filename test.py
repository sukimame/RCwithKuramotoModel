import numpy as np

def RandomInput(n, delay):
    seed = 42
    np.random.seed(seed)

    # Generate random input uniformly distribu
    data = np.random.uniform(0, 0.5, n + delay)
    return data[:-delay], data[delay:]

u, u2 = RandomInput(5, 1)

print(u, u2, sep="\n")