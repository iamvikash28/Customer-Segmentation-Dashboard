"""
=============================================================
  CUSTOMER SEGMENTATION PROJECT
  Sample Data Generator
=============================================================
  Generates realistic sample data so you can run all
  4 phases immediately without a real CRM.

  Output : data/crm_customers.csv
           data/orders.csv
           data/web_logs.csv
=============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

os.makedirs("data", exist_ok=True)
np.random.seed(42)
random.seed(42)

N_CUSTOMERS = 3050
START_DATE  = datetime(2022, 1, 1)
END_DATE    = datetime(2024, 12, 31)

print("=" * 55)
print("  GENERATING SAMPLE DATA")
print("=" * 55)

# ── 1. CRM CUSTOMERS ─────────────────────────────────────────────────────────
regions   = ["North", "South", "East", "West", "Central"]
sources   = ["Organic", "Paid", "Referral", "Social", "Email"]
cust_types = ["B2C", "B2B"]

customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS + 1)]
signup_dates = [START_DATE + timedelta(days=random.randint(0, 700)) for _ in range(N_CUSTOMERS)]

customers_df = pd.DataFrame({
    "customer_id":   customer_ids,
    "name":          [f"Customer {i}" for i in range(1, N_CUSTOMERS + 1)],
    "email":         [f"customer{i}@example.com" for i in range(1, N_CUSTOMERS + 1)],
    "region":        np.random.choice(regions, N_CUSTOMERS),
    "customer_type": np.random.choice(cust_types, N_CUSTOMERS, p=[0.75, 0.25]),
    "lead_source":   np.random.choice(sources, N_CUSTOMERS),
    "signup_date":   signup_dates,
})

# Introduce some missing regions (realistic noise)
null_idx = np.random.choice(N_CUSTOMERS, size=80, replace=False)
customers_df.loc[null_idx, "region"] = np.nan

customers_df.to_csv("data/crm_customers.csv", index=False)
print(f"[SAVED] data/crm_customers.csv  ({len(customers_df):,} rows)")


# ── 2. ORDERS ─────────────────────────────────────────────────────────────────
# Segment-based order patterns
seg_profiles = {
    "champion":        {"n_orders": (6, 18),  "amount": (80, 300),  "recency_cap": 40},
    "loyal":           {"n_orders": (4, 10),  "amount": (50, 200),  "recency_cap": 90},
    "potential":       {"n_orders": (2, 6),   "amount": (40, 150),  "recency_cap": 120},
    "new":             {"n_orders": (1, 3),   "amount": (30, 120),  "recency_cap": 30},
    "at_risk":         {"n_orders": (3, 9),   "amount": (50, 180),  "recency_cap": 300},
    "freq_low":        {"n_orders": (5, 14),  "amount": (10, 60),   "recency_cap": 60},
    "lost":            {"n_orders": (1, 4),   "amount": (20, 100),  "recency_cap": 500},
}

seg_weights = [0.16, 0.20, 0.18, 0.13, 0.10, 0.09, 0.14]
seg_names   = list(seg_profiles.keys())
assigned    = np.random.choice(seg_names, N_CUSTOMERS, p=seg_weights)

order_rows = []
order_id   = 1
statuses   = ["completed", "completed", "completed", "completed", "delivered", "returned"]

for i, cust_id in enumerate(customer_ids):
    profile = seg_profiles[assigned[i]]
    n = random.randint(*profile["n_orders"])
    latest = END_DATE - timedelta(days=random.randint(1, profile["recency_cap"]))

    for j in range(n):
        order_date = latest - timedelta(days=random.randint(j*10, j*30 + 30))
        if order_date < START_DATE:
            order_date = START_DATE + timedelta(days=random.randint(0, 30))
        amount = round(random.uniform(*profile["amount"]), 2)
        order_rows.append({
            "order_id":     f"O{str(order_id).zfill(6)}",
            "customer_id":  cust_id,
            "order_date":   order_date.strftime("%Y-%m-%d"),
            "order_amount": amount,
            "order_status": random.choice(statuses),
            "product_category": random.choice(["Electronics","Clothing","Home","Beauty","Sports","Books"]),
        })
        order_id += 1

orders_df = pd.DataFrame(order_rows)
orders_df.to_csv("data/orders.csv", index=False)
print(f"[SAVED] data/orders.csv          ({len(orders_df):,} rows)")


# ── 3. WEB LOGS ───────────────────────────────────────────────────────────────
log_rows  = []
session_id = 1
for cust_id in customer_ids:
    n_sessions = random.randint(1, 25)
    for _ in range(n_sessions):
        visit_date = START_DATE + timedelta(days=random.randint(0, 1095))
        log_rows.append({
            "session_id":  f"S{str(session_id).zfill(7)}",
            "customer_id": cust_id,
            "visit_date":  visit_date.strftime("%Y-%m-%d"),
            "page_views":  random.randint(1, 20),
        })
        session_id += 1

web_logs_df = pd.DataFrame(log_rows)
web_logs_df.to_csv("data/web_logs.csv", index=False)
print(f"[SAVED] data/web_logs.csv        ({len(web_logs_df):,} rows)")

print("\n  Sample data ready. Run phases 1–4 in order.")
print("=" * 55)
