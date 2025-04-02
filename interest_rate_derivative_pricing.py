import numpy as np
from scipy.stats import norm

# Simple discount factor from flat yield curve
def discount_factor(rate, t):
    return np.exp(-rate * t)

# Fixed-for-floating Interest Rate Swap pricing (payer)
def price_swap(notional, fixed_rate, float_rate, maturity, payments_per_year, discount_rate):
    dt = 1 / payments_per_year
    times = np.arange(dt, maturity + dt, dt)
    df = np.array([discount_factor(discount_rate, t) for t in times])

    fixed_leg = np.sum(fixed_rate * notional * dt * df)
    float_leg = np.sum(float_rate * notional * dt * df)  # assuming constant fwd rate

    swap_value = float_leg - fixed_leg
    print(f"💼 Interest Rate Swap Value: {swap_value:.4f}")
    return swap_value

# Black's model for cap/floorlet pricing
def black_caplet_price(F, K, T, sigma, r, notional, is_cap=True):
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    df = discount_factor(r, T)
    if is_cap:
        price = df * notional * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = df * notional * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    return price

# Price a full cap or floor
def price_cap_floor(notional, strike, forward_rate, maturity, payments_per_year, sigma, r, is_cap=True):
    dt = 1 / payments_per_year
    times = np.arange(dt, maturity + dt, dt)

    total = 0
    for T in times:
        price = black_caplet_price(forward_rate, strike, T, sigma, r, notional * dt, is_cap)
        total += price

    label = "Cap" if is_cap else "Floor"
    print(f"🎯 {label} Value: {total:.4f}")
    return total

# Test examples
def main():
    print("Interest Rate Derivative Pricing\n")

    # === Interest Rate Swap ===
    notional = 1_000_000
    fixed_rate = 0.04
    float_rate = 0.035
    maturity = 5  # years
    payments_per_year = 2
    discount_rate = 0.03

    price_swap(notional, fixed_rate, float_rate, maturity, payments_per_year, discount_rate)

    # === Cap and Floor ===
    strike = 0.04
    forward_rate = 0.035
    sigma = 0.2
    r = 0.03

    price_cap_floor(notional, strike, forward_rate, maturity, payments_per_year, sigma, r, is_cap=True)   # Cap
    price_cap_floor(notional, strike, forward_rate, maturity, payments_per_year, sigma, r, is_cap=False)  # Floor

if __name__ == "__main__":
    main()
