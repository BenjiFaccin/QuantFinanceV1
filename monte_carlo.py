import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_option_pricing(S, K, days, r, sigma, simulations=100000, steps=100):
    print("\n--- Monte Carlo Simulation for European Options ---")
    print(f"Spot Price (S): {S}")
    print(f"Strike Price (K): {K}")
    print(f"Days to Expiration: {days}")
    T = days / 360
    print(f"Converted Time to Maturity (T in years): {T:.6f}")
    print(f"Risk-free Rate (r): {r}")
    print(f"Volatility (sigma): {sigma}")
    print(f"Number of Simulations: {simulations}")
    print(f"Time Steps per Path: {steps}\n")

    dt = T / steps
    np.random.seed(42)

    # Generate price paths
    Z = np.random.standard_normal((simulations, steps))
    ST_paths = np.zeros_like(Z)
    ST_paths[:, 0] = S

    for t in range(1, steps):
        ST_paths[:, t] = ST_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])

    # Final simulated prices
    ST = ST_paths[:, -1]

    # Payoffs
    call_payoff = np.maximum(ST - K, 0)
    put_payoff = np.maximum(K - ST, 0)

    call_price = np.exp(-r * T) * np.mean(call_payoff)
    put_price = np.exp(-r * T) * np.mean(put_payoff)

    print(f"📈 Estimated Call Option Price: {call_price:.4f}")
    print(f"📉 Estimated Put Option Price:  {put_price:.4f}")

    # Plotting a sample of simulated paths
    plt.figure(figsize=(10, 6))
    for i in range(min(1000, simulations)):
        plt.plot(np.linspace(0, days, steps), ST_paths[i], lw=0.8, alpha=0.6)

    plt.title("Simulated Stock Price Paths")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

    return call_price, put_price

def main():
    print("Monte Carlo Option Pricing Model with Path Visualization\n")
    S = float(input("Enter the current stock price (S): "))
    K = float(input("Enter the strike price (K): "))
    days = int(input("Enter days to expiration: "))
    r = float(input("Enter the risk-free interest rate (r) in decimal (e.g. 0.05): "))
    sigma = float(input("Enter the volatility (sigma) in decimal (e.g. 0.2): "))
    simulations = int(input("Enter number of simulations (e.g. 100000): "))

    monte_carlo_option_pricing(S, K, days, r, sigma, simulations)

if __name__ == "__main__":
    main()
