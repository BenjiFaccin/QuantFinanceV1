import numpy as np

# =================== HESTON MODEL ===================

def heston_simulation(S0, v0, r, kappa, theta, sigma, rho, T, steps, n_paths):
    dt = T / steps
    S = np.zeros((n_paths, steps + 1))
    v = np.zeros((n_paths, steps + 1))

    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(1, steps + 1):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * np.random.standard_normal(n_paths)

        v[:, t] = np.maximum(v[:, t - 1] + kappa * (theta - v[:, t - 1]) * dt + sigma * np.sqrt(v[:, t - 1]) * np.sqrt(dt) * Z2, 0)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * v[:, t - 1]) * dt + np.sqrt(v[:, t - 1]) * np.sqrt(dt) * Z1)

    return S, v

def price_european_call_mc(S, K, r, T):
    payoff = np.maximum(S[:, -1] - K, 0)
    return np.exp(-r * T) * np.mean(payoff)


# =================== SABR MODEL ===================

def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    """Returns implied vol using Hagan's SABR approximation."""
    if F == K:
        term1 = ((1 - beta) ** 2 / 24) * (alpha ** 2) / (F ** (2 - 2 * beta))
        term2 = (rho * beta * nu * alpha) / (4 * F ** (1 - beta))
        term3 = (nu ** 2) * (2 - 3 * rho ** 2) / 24
        return alpha / (F ** (1 - beta)) * (1 + (term1 + term2 + term3) * T)

    logFK = np.log(F / K)
    FK_avg = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * FK_avg * logFK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

    one_beta = 1 - beta
    A = alpha / (FK_avg * (1 + one_beta ** 2 * logFK ** 2 / 24 + one_beta ** 4 * logFK ** 4 / 1920))
    B1 = (one_beta ** 2 * alpha ** 2) / (24 * (F * K) ** one_beta)
    B2 = (rho * beta * nu * alpha) / (4 * (F * K) ** ((1 - beta) / 2))
    B3 = (2 - 3 * rho ** 2) * nu ** 2 / 24
    B = 1 + (B1 + B2 + B3) * T

    return A * z / x_z * B


# =================== COMBINED INTERFACE ===================

def main():
    print("🔁 Heston + SABR Model Toolkit\n")
    print("Select model:")
    print("1 - Heston Monte Carlo (European Call)")
    print("2 - SABR Implied Volatility")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        # HESTON PRICING
        print("\n=== Heston Monte Carlo ===")
        S0 = float(input("Initial stock price (S0): "))
        K = float(input("Strike price (K): "))
        days = int(input("Days to expiration: "))
        T = days / 365
        r = float(input("Risk-free rate (r): "))
        v0 = float(input("Initial variance (v0): "))
        kappa = float(input("Mean reversion speed (kappa): "))
        theta = float(input("Long-term variance (theta): "))
        sigma = float(input("Vol of variance (sigma): "))
        rho = float(input("Correlation (rho, e.g. -0.7): "))
        steps = int(input("Time steps (e.g. 100): "))
        n_paths = int(input("Simulation paths (e.g. 10000): "))

        np.random.seed(42)
        S, v = heston_simulation(S0, v0, r, kappa, theta, sigma, rho, T, steps, n_paths)
        price = price_european_call_mc(S, K, r, T)

        print(f"\n📈 Heston Monte Carlo Call Price: {price:.4f}")

    elif choice == "2":
        # SABR VOL
        print("\n=== SABR Implied Volatility ===")
        F = float(input("Forward rate (F): "))
        K = float(input("Strike rate (K): "))
        days = int(input("Days to expiration: "))
        T = days / 365
        alpha = float(input("Alpha (initial vol): "))
        beta = float(input("Beta (0 to 1): "))
        rho = float(input("Rho (correlation): "))
        nu = float(input("Nu (vol of vol): "))

        iv = sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
        print(f"\n📉 SABR Implied Volatility: {iv:.4%}")

    else:
        print("Invalid choice. Run the program again.")

if __name__ == "__main__":
    main()
