import numpy as np
import matplotlib.pyplot as plt

def generate_multifractal_time(n, H=0.5, lambda2=0.2):
    """
    Generates multifractal trading time using a log-normal cascade
    """
    log_vol = np.random.normal(loc=-lambda2 / 2, scale=np.sqrt(lambda2), size=n)
    weights = np.exp(log_vol)
    weights /= np.sum(weights)
    multifractal_time = np.cumsum(weights)
    multifractal_time /= multifractal_time[-1]  # normalize to [0,1]
    return multifractal_time

def simulate_mmar(S0=100, T=1, steps=1000, n_paths=5, H=0.5, lambda2=0.2):
    dt = T / steps
    time_grid = np.linspace(0, T, steps + 1)
    all_paths = []

    for _ in range(n_paths):
        # Step 1: Generate multifractal time
        theta = generate_multifractal_time(steps + 1, H=H, lambda2=lambda2)

        # Step 2: Simulate Brownian motion
        dW = np.random.normal(0, np.sqrt(np.diff(theta)))
        W = np.concatenate([[0], np.cumsum(dW)])

        # Step 3: Price process
        log_returns = W  # MMAR: log(S) ~ B(θ(t))
        S = S0 * np.exp(log_returns - 0.5 * np.var(log_returns))  # adjust for drift
        all_paths.append(S)

    return time_grid, np.array(all_paths)

def plot_paths(time_grid, paths):
    plt.figure(figsize=(10, 6))
    for path in paths:
        plt.plot(time_grid, path, lw=1.5, alpha=0.8)
    plt.title("Multifractal Monte Carlo Simulation of Asset Paths")
    plt.xlabel("Time")
    plt.ylabel("Asset Price")
    plt.grid(True)
    plt.show()

def main():
    print("📈 Multifractal Monte Carlo Simulation (MMAR)\n")
    S0 = float(input("Initial stock price (S0): "))
    days = int(input("Simulation horizon in days (e.g. 252): "))
    steps = int(input("Number of time steps: "))
    n_paths = int(input("Number of paths to simulate: "))
    H = float(input("Hurst exponent H (0.5 = Brownian, < 0.5 = anti-persistent): "))
    lambda2 = float(input("Intermittency parameter λ² (e.g. 0.2): "))

    T = days / 365  # Convert days to years
    time_grid, paths = simulate_mmar(S0, T, steps, n_paths, H, lambda2)
    plot_paths(time_grid, paths)

if __name__ == "__main__":
    main()
