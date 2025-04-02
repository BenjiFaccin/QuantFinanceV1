import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# === Black-Scholes Call Pricing & Delta ===
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return price, delta

# === Delta Hedging Simulator ===
def delta_hedging_simulation(S0, K, T_days, r, sigma, dt_days, n_steps):
    T = T_days / 365
    dt = dt_days / 365
    steps = int(T / dt)

    # Initialize arrays
    stock_prices = [S0]
    option_prices = []
    deltas = []
    hedge_positions = []
    cash_positions = []

    S = S0
    option_price, delta = black_scholes_call(S, K, T, r, sigma)
    option_prices.append(option_price)
    deltas.append(delta)

    hedge = -delta
    hedge_positions.append(hedge)
    cash = option_price + hedge * S
    cash_positions.append(cash)

    for i in range(1, steps + 1):
        # Simulate next stock price (random walk)
        dW = np.random.normal(0, np.sqrt(dt))
        S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW)
        stock_prices.append(S)

        T_left = T - i * dt
        option_price, delta = black_scholes_call(S, K, T_left, r, sigma)
        option_prices.append(option_price)
        deltas.append(delta)

        hedge = -delta
        hedge_positions.append(hedge)

        cash = cash_positions[-1] * np.exp(r * dt) + (hedge_positions[-2] - hedge) * S
        cash_positions.append(cash)

    # Final PnL
    final_value = hedge * S + cash + option_prices[-1]
    pnl = final_value - option_prices[0]

    print(f"\n📉 Final PnL of Hedged Portfolio: {pnl:.4f}")

    return stock_prices, option_prices, deltas, cash_positions

# === Visualization ===
def plot_results(stock_prices, deltas, cash_positions):
    time = np.arange(len(stock_prices))
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax[0].plot(time, stock_prices)
    ax[0].set_ylabel("Stock Price")
    ax[0].grid(True)

    ax[1].plot(time, deltas)
    ax[1].set_ylabel("Option Delta")
    ax[1].grid(True)

    ax[2].plot(time, cash_positions)
    ax[2].set_ylabel("Cash Position")
    ax[2].set_xlabel("Time Steps")
    ax[2].grid(True)

    plt.suptitle("Delta Hedging Strategy Over Time")
    plt.show()

# === Main Interface ===
def main():
    print("Portfolio Hedging Strategy — Delta Hedging Simulator\n")

    S0 = float(input("Initial stock price: "))
    K = float(input("Option strike price: "))
    T_days = int(input("Days to expiration: "))
    r = float(input("Risk-free rate (e.g. 0.01): "))
    sigma = float(input("Volatility (e.g. 0.2): "))
    dt_days = int(input("Rebalancing frequency in days (e.g. 1): "))

    n_steps = int(T_days / dt_days)

    stock_prices, option_prices, deltas, cash_positions = delta_hedging_simulation(
        S0, K, T_days, r, sigma, dt_days, n_steps
    )

    plot_results(stock_prices, deltas, cash_positions)

if __name__ == "__main__":
    main()
