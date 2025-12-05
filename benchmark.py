import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from src.config import PARAMS, N_PATHS_TRAIN, N_PATHS_TEST, INNER_PATHS, MPOR
from src.g2_model import G2PlusPlus
from src.nested_mc import calculate_dim_nested
from src.neural_dim import DIMNet, train_dim_model


def main():
    model_g2 = G2PlusPlus(PARAMS)

    # 1. Data Generation
    print(f"Generating training data ({N_PATHS_TRAIN} scenarios)...")
    t_train = np.random.uniform(0, 10, N_PATHS_TRAIN)
    x_train = np.random.normal(0, 0.05, N_PATHS_TRAIN)
    y_train = np.random.normal(0, 0.05, N_PATHS_TRAIN)

    states_train = {'t': t_train, 'x': x_train, 'y': y_train}

    dim_labels = calculate_dim_nested(model_g2, states_train, MPOR, n_nested=INNER_PATHS)

    X_train = torch.tensor(np.column_stack((t_train, x_train, y_train)), dtype=torch.float32)
    Y_train = torch.tensor(dim_labels.reshape(-1, 1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=256, shuffle=True)

    # 2. Training
    print("Training Deep Surrogate...")
    net = DIMNet()
    train_dim_model(net, loader, epochs=100)

    # 3. Benchmark
    print(f"Benchmarking on {N_PATHS_TEST} scenarios...")

    t_test = np.random.uniform(0, 10, N_PATHS_TEST)
    x_test = np.random.normal(0, 0.05, N_PATHS_TEST)
    y_test = np.random.normal(0, 0.05, N_PATHS_TEST)
    states_test = {'t': t_test, 'x': x_test, 'y': y_test}

    # Nested MC
    t0 = time.time()
    dim_mc = calculate_dim_nested(model_g2, states_test, MPOR, n_nested=INNER_PATHS)
    time_mc = time.time() - t0

    # Neural Net
    t0 = time.time()
    net.eval()
    with torch.no_grad():
        inputs = torch.tensor(np.column_stack((t_test, x_test, y_test)), dtype=torch.float32)
        dim_nn = net(inputs).numpy().flatten()
    time_nn = time.time() - t0

    print(f"Nested MC Time: {time_mc:.4f}s")
    print(f"Neural Net Time: {time_nn:.4f}s")
    print(f"Speedup: x{time_mc / time_nn:.1f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(dim_mc, dim_nn, alpha=0.5, s=5)
    plt.plot([min(dim_mc), max(dim_mc)], [min(dim_mc), max(dim_mc)], 'r--')
    plt.xlabel("Nested MC")
    plt.ylabel("Neural Net")
    plt.title(f"DIM Approximation: Deep Learning vs Nested MC\nSpeedup: x{time_mc / time_nn:.0f}")
    plt.savefig("dim_benchmark.png")


if __name__ == "__main__":
    main()