import numpy as np

def generate_price_paths(S0, r, sigma, T, steps, n_paths):
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0

    Z = np.random.standard_normal((n_paths, steps))
    for t in range(1, steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])
    return paths

def price_asian_call(paths, K, r, T):
    avg_price = np.mean(paths[:, 1:], axis=1)
    payoff = np.maximum(avg_price - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

def price_barrier_call(paths, K, B, r, T):
    hit_barrier = np.any(paths >= B, axis=1)
    final_price = paths[:, -1]
    payoff = np.where(~hit_barrier, np.maximum(final_price - K, 0), 0)
    return np.exp(-r * T) * np.mean(payoff)

def price_lookback_call(paths, r, T):
    min_price = np.min(paths[:, 1:], axis=1)
    final_price = paths[:, -1]
    payoff = final_price - min_price
    return np.exp(-r * T) * np.mean(payoff)

def main():
    print("Exotic Option Pricing (Monte Carlo) — With Days to Expiration\n")

    S0 = float(input("Initial stock price (S0): "))
    K = float(input("Strike price (K): "))
    B = float(input("Barrier level (for knock-out call): "))
    days = int(input("Days to expiration: "))
    r = float(input("Risk-free rate (e.g. 0.05): "))
    sigma = float(input("Volatility (sigma, e.g. 0.2): "))
    steps = int(input("Number of time steps: "))
    n_paths = int(input("Number of simulation paths: "))

    T = days / 365  # Convert days to years

    print("\nSimulating paths and pricing options...")

    np.random.seed(42)
    paths = generate_price_paths(S0, r, sigma, T, steps, n_paths)

    asian = price_asian_call(paths, K, r, T)
    barrier = price_barrier_call(paths, K, B, r, T)
    lookback = price_lookback_call(paths, r, T)

    print(f"\n📊 Asian Call Option Price:     {asian:.4f}")
    print(f"🚧 Barrier Knock-Out Call Price: {barrier:.4f}")
    print(f"🔍 Lookback Call Option Price:  {lookback:.4f}")

if __name__ == "__main__":
    main()
