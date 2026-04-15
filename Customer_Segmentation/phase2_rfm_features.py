import pandas as pd
import numpy as np
from datetime import datetime

# ── 1. LOAD MASTER TABLE FROM PHASE 1 ────────────────────────────────────────
master_df = pd.read_csv("output/customer_master.csv", parse_dates=[
    "last_order_date", "first_order_date", "signup_date"
])

orders_df = pd.read_csv("data/orders.csv", parse_dates=["order_date"])
orders_df = orders_df[orders_df["order_status"] == "completed"]  # confirmed orders only

print(f"Customers loaded: {len(master_df)}")


# ── 2. SET REFERENCE DATE (snapshot date for recency) ────────────────────────
# Use today, or set a fixed date for reproducibility
SNAPSHOT_DATE = pd.Timestamp("today").normalize()
# SNAPSHOT_DATE = pd.Timestamp("2024-12-31")  # uncomment for fixed date


# ── 3. CALCULATE CORE RFM METRICS ─────────────────────────────────────────────
rfm = orders_df.groupby("customer_id").agg(
    last_order_date  = ("order_date",   "max"),
    frequency        = ("order_id",     "count"),
    monetary         = ("order_amount", "sum")
).reset_index()

# Recency: days between last order and snapshot
rfm["recency_days"] = (SNAPSHOT_DATE - rfm["last_order_date"]).dt.days


# ── 4. EXTENDED FEATURES ──────────────────────────────────────────────────────
# Average order value
rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]

# Customer lifespan in days (first to last order)
order_span = orders_df.groupby("customer_id").agg(
    first_order_date = ("order_date", "min"),
    last_order_date  = ("order_date", "max")
).reset_index()

order_span["lifespan_days"] = (
    order_span["last_order_date"] - order_span["first_order_date"]
).dt.days

rfm = rfm.merge(order_span[["customer_id", "first_order_date", "lifespan_days"]],
                on="customer_id", how="left")

# Purchase velocity: orders per month (avoid division by zero)
rfm["orders_per_month"] = rfm.apply(
    lambda r: round(r["frequency"] / max(r["lifespan_days"] / 30, 1), 2), axis=1
)

# Simple CLV estimate: monetary × (12 / max(recency_days/30, 1))
rfm["estimated_clv"] = rfm["monetary"] * (12 / rfm["recency_days"].clip(lower=1) * 30)
rfm["estimated_clv"] = rfm["estimated_clv"].clip(upper=rfm["monetary"] * 10)  # cap outliers


# ── 5. RFM SCORING (1–5 scale using quintiles) ────────────────────────────────
# Recency: lower days = higher score (invert rank)
rfm["r_score"] = pd.qcut(rfm["recency_days"], q=5,
                          labels=[5, 4, 3, 2, 1]).astype(int)

# Frequency: higher count = higher score
rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5,
                          labels=[1, 2, 3, 4, 5]).astype(int)

# Monetary: higher spend = higher score
rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5,
                          labels=[1, 2, 3, 4, 5]).astype(int)

# Combined RFM score (string for segmentation rules) and numeric total
rfm["rfm_score"]   = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
rfm["rfm_total"]   = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]


# ── 6. SEGMENT LABELS BASED ON RFM SCORES ────────────────────────────────────
def assign_segment(row):
    r, f, m = row["r_score"], row["f_score"], row["m_score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3 and m >= 3:
        return "Loyal customers"
    elif r >= 4 and f <= 2:
        return "New customers"
    elif r >= 3 and f >= 3 and m <= 2:
        return "Frequent low-spenders"
    elif r <= 2 and f >= 3 and m >= 3:
        return "At-risk customers"
    elif r <= 2 and f >= 4 and m >= 4:
        return "Cant lose them"
    elif r <= 2 and f <= 2 and m <= 2:
        return "Lost / dormant"
    else:
        return "Potential loyalists"

rfm["segment"] = rfm.apply(assign_segment, axis=1)


# ── 7. MERGE BACK TO MASTER TABLE ────────────────────────────────────────────
feature_cols = [
    "customer_id", "recency_days", "frequency", "monetary",
    "avg_order_value", "lifespan_days", "orders_per_month",
    "estimated_clv", "r_score", "f_score", "m_score",
    "rfm_score", "rfm_total", "segment"
]

feature_matrix = master_df.merge(rfm[feature_cols], on="customer_id", how="left")

# Customers with no orders get zeroed out, segment = "No purchase"
if "segment" not in feature_matrix.columns:
    feature_matrix["segment"] = "No purchase"
else:
    feature_matrix["segment"] = feature_matrix["segment"].fillna("No purchase")

# ── 8. SUMMARY STATS BY SEGMENT ───────────────────────────────────────────────
print("\n=== SEGMENT SUMMARY ===")
summary = feature_matrix.groupby("segment").agg(
    count          = ("customer_id",   "count"),
    avg_recency    = ("recency_days",  "mean"),
    avg_frequency  = ("frequency",     "mean"),
    avg_monetary   = ("monetary",      "mean"),
    total_revenue  = ("monetary",      "sum")
).round(1)

print(summary.sort_values("total_revenue", ascending=False))


# ── 9. EXPORT ──────────────────────────────────────────────────────────────────
feature_matrix.to_csv("output/feature_matrix.csv", index=False)

with pd.ExcelWriter("output/rfm_analysis.xlsx", engine="openpyxl") as writer:
    feature_matrix.to_excel(writer, sheet_name="Feature Matrix", index=False)
    summary.to_excel(writer, sheet_name="Segment Summary")
    rfm[["customer_id", "r_score", "f_score", "m_score",
         "rfm_score", "rfm_total", "segment"]].to_excel(
        writer, sheet_name="RFM Scores", index=False)

print("\nPhase 2 complete. Files saved to /output/")