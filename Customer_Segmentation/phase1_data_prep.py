import os

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

import pandas as pd
import numpy as np
from datetime import datetime

# ── 1. LOAD RAW DATA ──────────────────────────────────────────────────────────
# Replace these paths with your actual files
customers_df  = pd.read_csv("data/crm_customers.csv")
orders_df     = pd.read_csv("data/orders.csv")
web_logs_df   = pd.read_csv("data/web_logs.csv")

print("Raw shapes:", customers_df.shape, orders_df.shape, web_logs_df.shape)


# ── 2. STANDARDISE COLUMN NAMES ───────────────────────────────────────────────
customers_df.columns = customers_df.columns.str.strip().str.lower().str.replace(" ", "_")
orders_df.columns    = orders_df.columns.str.strip().str.lower().str.replace(" ", "_")
web_logs_df.columns  = web_logs_df.columns.str.strip().str.lower().str.replace(" ", "_")


# ── 3. PARSE DATES ────────────────────────────────────────────────────────────
orders_df["order_date"]       = pd.to_datetime(orders_df["order_date"], errors="coerce")
customers_df["signup_date"]   = pd.to_datetime(customers_df["signup_date"], errors="coerce")


# ── 4. REMOVE DUPLICATES ──────────────────────────────────────────────────────
customers_df = customers_df.drop_duplicates(subset="customer_id")
orders_df    = orders_df.drop_duplicates(subset="order_id")


# ── 5. HANDLE MISSING VALUES ──────────────────────────────────────────────────
# Drop rows with no customer_id (unusable)
customers_df = customers_df.dropna(subset=["customer_id"])
orders_df    = orders_df.dropna(subset=["customer_id", "order_date", "order_amount"])

# Fill optional fields
customers_df["region"]  = customers_df["region"].fillna("Unknown")
# Create segment column if it doesn't exist
if "segment" not in customers_df.columns:
    customers_df["segment"] = "Unassigned"
else:
    customers_df["segment"] = customers_df["segment"].fillna("Unassigned")

# ── 6. REMOVE INVALID ORDERS ─────────────────────────────────────────────────
orders_df = orders_df[orders_df["order_amount"] > 0]


# ── 7. AGGREGATE WEB ENGAGEMENT ──────────────────────────────────────────────
web_summary = web_logs_df.groupby("customer_id").agg(
    total_sessions  = ("session_id",   "count"),
    total_pageviews = ("page_views",   "sum"),
    last_visit_date = ("visit_date",   "max")
).reset_index()

web_summary["last_visit_date"] = pd.to_datetime(web_summary["last_visit_date"])


# ── 8. BUILD ORDER SUMMARY PER CUSTOMER ──────────────────────────────────────
order_summary = orders_df.groupby("customer_id").agg(
    total_orders      = ("order_id",     "count"),
    total_revenue     = ("order_amount", "sum"),
    avg_order_value   = ("order_amount", "mean"),
    first_order_date  = ("order_date",   "min"),
    last_order_date   = ("order_date",   "max")
).reset_index()


# ── 9. MERGE EVERYTHING INTO MASTER TABLE ────────────────────────────────────
master_df = customers_df.merge(order_summary, on="customer_id", how="left")
master_df = master_df.merge(web_summary,      on="customer_id", how="left")

# Fill customers with no orders yet
master_df["total_orders"]    = master_df["total_orders"].fillna(0).astype(int)
master_df["total_revenue"]   = master_df["total_revenue"].fillna(0)
master_df["avg_order_value"] = master_df["avg_order_value"].fillna(0)
master_df["total_sessions"]  = master_df["total_sessions"].fillna(0).astype(int)


# ── 10. DATA QUALITY REPORT ───────────────────────────────────────────────────
print("\n=== DATA QUALITY REPORT ===")
print(f"Total customers     : {len(master_df)}")
print(f"Customers with orders: {(master_df['total_orders'] > 0).sum()}")
print(f"Missing values:\n{master_df.isnull().sum()[master_df.isnull().sum() > 0]}")
print(f"\nDate range: {orders_df['order_date'].min()} → {orders_df['order_date'].max()}")
print(f"Revenue range: ${orders_df['order_amount'].min():.2f} → ${orders_df['order_amount'].max():.2f}")


# ── 11. EXPORT ────────────────────────────────────────────────────────────────
master_df.to_csv("output/customer_master.csv", index=False)

# Also save quality report to Excel
with pd.ExcelWriter("output/data_quality_report.xlsx", engine="openpyxl") as writer:
    master_df.describe(include="all").to_excel(writer, sheet_name="Summary Stats")
    master_df.isnull().sum().to_frame("null_count").to_excel(writer, sheet_name="Null Counts")
    master_df.dtypes.to_frame("dtype").to_excel(writer, sheet_name="Data Types")

print("\nPhase 1 complete. Files saved to /output/")