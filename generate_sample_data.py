"""
generate_sample_data.py

Generates a realistic, reproducible synthetic sales dataset for demoing
InsightAI. Deliberately bakes in patterns for the analytics engine to find:

  - Bangalore is the strongest-performing region overall.
  - Pune shows a clear revenue decline in the final quarter of the period.
  - Mouse demand increases noticeably over time.
  - Webcam has a slowly declining profit margin (rising cost, flat price).
  - Mild seasonal variation (a November/December bump).

Run with: python generate_sample_data.py
Output:   data/sample_sales.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
NUM_ROWS = 1200
START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2025-06-30")

PRODUCTS = {
    # product: (category, base_unit_price, base_unit_cost)
    "Laptop": ("Computers", 55000, 42000),
    "Monitor": ("Computers", 12000, 8500),
    "Keyboard": ("Accessories", 1500, 900),
    "Mouse": ("Accessories", 800, 450),
    "Headphones": ("Peripherals", 2500, 1600),
    "Webcam": ("Peripherals", 3200, 1800),
}

REGIONS = ["Mumbai", "Pune", "Delhi", "Bangalore", "Hyderabad"]
REGION_WEIGHTS = [0.20, 0.20, 0.20, 0.25, 0.15]  # Bangalore weighted slightly higher

SALESPEOPLE = [
    "Aarav Shah", "Ishita Rao", "Rohan Mehta", "Priya Nair", "Karan Verma",
    "Sneha Iyer", "Aditya Kapoor", "Neha Joshi",
]

CUSTOMER_TYPES = ["Retail", "Corporate", "Distributor"]
CUSTOMER_WEIGHTS = [0.55, 0.30, 0.15]


def generate_dataset(num_rows: int = NUM_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    date_range_days = (END_DATE - START_DATE).days
    random_offsets = rng.integers(0, date_range_days + 1, size=num_rows)
    dates = [START_DATE + pd.Timedelta(days=int(offset)) for offset in random_offsets]

    products = rng.choice(list(PRODUCTS.keys()), size=num_rows)
    regions = rng.choice(REGIONS, size=num_rows, p=REGION_WEIGHTS)
    salespeople = rng.choice(SALESPEOPLE, size=num_rows)
    customer_types = rng.choice(CUSTOMER_TYPES, size=num_rows, p=CUSTOMER_WEIGHTS)

    rows = []
    for i in range(num_rows):
        product = products[i]
        category, base_price, base_cost = PRODUCTS[product]
        date = dates[i]
        region = regions[i]

        # --- Baked-in pattern: Mouse demand increases over time ---
        month_index = (date.year - START_DATE.year) * 12 + (date.month - START_DATE.month)
        mouse_boost = 1.0
        if product == "Mouse":
            mouse_boost = 1.0 + min(month_index * 0.05, 1.2)

        # --- Baked-in pattern: Pune decline in the final quarter of the period ---
        pune_decline = 1.0
        if region == "Pune" and date >= pd.Timestamp("2025-04-01"):
            pune_decline = 0.55

        # --- Seasonal bump: Nov/Dec ---
        seasonal_boost = 1.25 if date.month in (11, 12) else 1.0

        # --- Baked-in pattern: Webcam margin slowly declines (cost rises faster than price) ---
        cost_drift = 1.0
        if product == "Webcam":
            cost_drift = 1.0 + min(month_index * 0.02, 0.35)

        base_units = rng.poisson(lam=4) + 1
        units_sold = max(1, int(base_units * mouse_boost * pune_decline * seasonal_boost))

        price_noise = rng.normal(1.0, 0.03)
        unit_price = round(base_price * price_noise, 2)

        cost_noise = rng.normal(1.0, 0.03)
        unit_cost = round(base_cost * cost_drift * cost_noise, 2)

        revenue = round(units_sold * unit_price, 2)
        cost = round(units_sold * unit_cost, 2)
        profit = round(revenue - cost, 2)

        rows.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Order_ID": f"ORD{100000 + i}",
            "Product": product,
            "Category": category,
            "Region": region,
            "Salesperson": salespeople[i],
            "Units_Sold": units_sold,
            "Unit_Price": unit_price,
            "Revenue": revenue,
            "Cost": cost,
            "Profit": profit,
            "Customer_Type": customer_types[i],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def main():
    df = generate_dataset()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample_sales.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows -> {output_path}")


if __name__ == "__main__":
    main()
