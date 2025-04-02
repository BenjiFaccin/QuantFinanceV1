import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import brentq
from scipy.stats import norm

# Black-Scholes Call Option Pricing Formula
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Implied volatility via root-finding
def implied_volatility_call(C_market, S, K, T, r):
    try:
        return brentq(lambda sigma: black_scholes_call_price(S, K, T, r, sigma) - C_market, 1e-6, 3.0)
    except ValueError:
        return np.nan

# Generate volatility surface from sample option price grid
def build_vol_surface(option_data, S, r):
    strikes = sorted(set([d['K'] for d in option_data]))
    maturities = sorted(set([d['T'] for d in option_data]))

    IV_surface = np.zeros((len(maturities), len(strikes)))

    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            match = next((d for d in option_data if d['K'] == K and d['T'] == T), None)
            if match:
                IV_surface[i, j] = implied_volatility_call(match['C'], S, K, T, r)
            else:
                IV_surface[i, j] = np.nan

    return np.array(strikes), np.array(maturities), IV_surface

# Plot surface
def plot_vol_surface(strikes, maturities, IV_surface):
    X, Y = np.meshgrid(strikes, maturities)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, IV_surface, cmap='viridis', edgecolor='k', linewidth=0.5)
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Maturity (Years)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Volatility Surface')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# Sample test case
def main():
    print("Volatility Surface Construction and Visualization\n")

    # Example data: Option prices across strikes and maturities
    # Replace with your real market data
    S = 100  # Spot price
    r = 0.05

    option_data = [
        {"K": 90,  "T": 0.25, "C": 12.5},
        {"K": 100, "T": 0.25, "C": 7.3},
        {"K": 110, "T": 0.25, "C": 3.9},
        {"K": 90,  "T": 0.5,  "C": 14.0},
        {"K": 100, "T": 0.5,  "C": 9.2},
        {"K": 110, "T": 0.5,  "C": 5.5},
        {"K": 90,  "T": 1.0,  "C": 16.8},
        {"K": 100, "T": 1.0,  "C": 12.0},
        {"K": 110, "T": 1.0,  "C": 8.1},
    ]

    strikes, maturities, IV_surface = build_vol_surface(option_data, S, r)
    plot_vol_surface(strikes, maturities, IV_surface)

if __name__ == "__main__":
    main()
#test