import numpy as np

def binomial_tree_american_option(S, K, days, r, sigma, steps=100, option_type='call', dividend_schedule=[]):
    print("\n--- Binomial Tree American Option Pricing ---")
    T = days / 360
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)

    print(f"Time to maturity (T): {T:.6f} years")
    print(f"Step size (dt): {dt:.6f}")
    print(f"Up factor (u): {u:.4f}")
    print(f"Down factor (d): {d:.4f}")
    print(f"Risk-neutral prob. (q): {q:.4f}\n")

    # Adjust dividend steps to tree step indexes
    dividend_steps = [int(t / T * steps) for t in dividend_schedule]

    # Build stock price tree
    stock_tree = np.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)
    
    # Subtract dividends at the right steps
    for div_step in dividend_steps:
        for j in range(div_step + 1):
            stock_tree[j, div_step] = max(stock_tree[j, div_step] - dividend_schedule[dividend_steps.index(div_step + 1)], 0)

    # Build option value tree
    option_tree = np.zeros_like(stock_tree)

    # Terminal payoffs
    for j in range(steps + 1):
        if option_type == 'call':
            option_tree[j, steps] = max(stock_tree[j, steps] - K, 0)
        else:
            option_tree[j, steps] = max(K - stock_tree[j, steps], 0)

    # Backward induction
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            hold = np.exp(-r * dt) * (q * option_tree[j, i + 1] + (1 - q) * option_tree[j + 1, i + 1])
            exercise = (stock_tree[j, i] - K if option_type == 'call' else K - stock_tree[j, i])
            option_tree[j, i] = max(hold, exercise)

    print(f"📈 American {option_type.capitalize()} Option Price: {option_tree[0,0]:.4f}")
    return option_tree[0,0]

def main():
    print("American Option Pricing using Binomial Tree (with Dividends)\n")
    S = float(input("Enter the current stock price (S): "))
    K = float(input("Enter the strike price (K): "))
    days = int(input("Enter days to expiration: "))
    r = float(input("Enter the risk-free rate (e.g. 0.05): "))
    sigma = float(input("Enter volatility (e.g. 0.2): "))
    steps = int(input("Enter number of steps in the tree (e.g. 100): "))
    option_type = input("Option type ('call' or 'put'): ").lower()
    
    # Discrete dividends
    div_count = int(input("Number of dividend payments before expiration: "))
    dividend_schedule = []
    for i in range(div_count):
        t = float(input(f"  Enter time (in years) of dividend {i+1}: "))
        amount = float(input(f"  Enter dividend amount {i+1}: "))
        dividend_schedule.append((t, amount))

    # Flatten dividend times
    div_times = [t for t, _ in dividend_schedule]
    div_values = [v for _, v in dividend_schedule]

    binomial_tree_american_option(S, K, days, r, sigma, steps, option_type, div_values)

if __name__ == "__main__":
    main()
