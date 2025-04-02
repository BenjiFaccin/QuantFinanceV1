import numpy as np

def discount_factor(rate, t):
    return np.exp(-rate * t)

def cds_pricing(notional, maturity, r, recovery_rate, hazard_rate, payments_per_year):
    dt = 1 / payments_per_year
    times = np.arange(dt, maturity + dt, dt)
    
    # --- Survival and default probabilities
    survival_probs = np.exp(-hazard_rate * times)
    default_probs = np.zeros_like(times)
    default_probs[0] = 1 - survival_probs[0]
    for i in range(1, len(times)):
        default_probs[i] = survival_probs[i-1] - survival_probs[i]

    # --- Discount factors
    dfs = np.array([discount_factor(r, t) for t in times])

    # --- Premium leg: payments made if no default
    premium_leg = np.sum(dfs * survival_probs * dt) * notional

    # --- Protection leg: payment on default
    protection_leg = np.sum(dfs * default_probs) * notional * (1 - recovery_rate)

    # --- Fair spread
    fair_spread = protection_leg / premium_leg

    print("\n--- CDS Pricing Summary ---")
    print(f"Notional: {notional}")
    print(f"Maturity: {maturity} years")
    print(f"Risk-Free Rate: {r}")
    print(f"Recovery Rate: {recovery_rate}")
    print(f"Hazard Rate (λ): {hazard_rate}")
    print(f"Payments per Year: {payments_per_year}")
    print(f"\n📈 Fair CDS Spread: {fair_spread * 10000:.2f} bps")

    return fair_spread

def main():
    print("Credit Default Swap (CDS) Fair Spread Calculator\n")
    notional = float(input("Enter notional amount: "))
    maturity = float(input("Enter CDS maturity (in years): "))
    r = float(input("Enter constant risk-free rate (e.g. 0.03): "))
    recovery_rate = float(input("Enter recovery rate (e.g. 0.4): "))
    hazard_rate = float(input("Enter constant hazard rate (e.g. 0.02): "))
    payments_per_year = int(input("Payments per year (e.g. 4 for quarterly): "))

    cds_pricing(notional, maturity, r, recovery_rate, hazard_rate, payments_per_year)

if __name__ == "__main__":
    main()
