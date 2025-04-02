import math
from scipy.stats import norm

# Black-Scholes formula for European Call and Put
def black_scholes(S, K, days, r, sigma):
    print("\n--- Black-Scholes Calculation Steps ---")
    print(f"Spot Price (S): {S}")
    print(f"Strike Price (K): {K}")
    print(f"Days to Expiration: {days}")
    T = days / 360
    print(f"Converted Time to Maturity (T in years): {T:.6f}")
    print(f"Risk-free Rate (r): {r}")
    print(f"Volatility (sigma): {sigma}\n")

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    print(f"Calculated d1: {d1:.4f}")
    print(f"Calculated d2: {d2:.4f}")
    print(f"N(d1): {norm.cdf(d1):.4f}")
    print(f"N(d2): {norm.cdf(d2):.4f}")
    print(f"N(-d1): {norm.cdf(-d1):.4f}")
    print(f"N(-d2): {norm.cdf(-d2):.4f}\n")

    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    print(f"📈 Call Option Price: {call_price:.4f}")
    print(f"📉 Put Option Price:  {put_price:.4f}")

    return call_price, put_price

# Gather inputs from the user
def main():
    print("Black-Scholes Option Pricing Model\n")
    S = float(input("Enter the current stock price (S): "))
    K = float(input("Enter the strike price (K): "))
    days = int(input("Enter days to expiration: "))
    r = float(input("Enter the risk-free interest rate (r) in decimal (e.g. 0.05): "))
    sigma = float(input("Enter the volatility (sigma) in decimal (e.g. 0.2): "))

    black_scholes(S, K, days, r, sigma)

if __name__ == "__main__":
    main()
