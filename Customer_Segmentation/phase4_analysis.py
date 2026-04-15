import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD LABELED SEGMENTS FROM PHASE 3 ────────────────────────────────────
df = pd.read_csv("output/segments_labeled.csv")
df = df[df["frequency"] > 0].copy()

SEGMENT_COL = "segment_label"
FEATURES    = ["recency_days", "frequency", "monetary",
               "avg_order_value", "orders_per_month", "lifespan_days"]

# Consistent color palette — one color per segment
SEGMENTS    = df[SEGMENT_COL].unique()
PALETTE     = dict(zip(SEGMENTS, [
    "#378ADD", "#1D9E75", "#EF9F27",
    "#D85A30", "#7F77DD", "#D4537E", "#639922", "#888780"
][:len(SEGMENTS)]))

print(f"Segments found: {list(SEGMENTS)}")
print(f"Total customers: {len(df)}\n")


# ── 2. SEGMENT SIZING ─────────────────────────────────────────────────────────
sizing = df.groupby(SEGMENT_COL).agg(
    customer_count = ("customer_id",   "count"),
    total_revenue  = ("monetary",      "sum"),
    avg_revenue    = ("monetary",      "mean"),
    avg_recency    = ("recency_days",  "mean"),
    avg_frequency  = ("frequency",     "mean"),
    avg_aov        = ("avg_order_value","mean"),
).round(2)

sizing["revenue_share_pct"] = (
    sizing["total_revenue"] / sizing["total_revenue"].sum() * 100
).round(1)

sizing["customer_share_pct"] = (
    sizing["customer_count"] / sizing["customer_count"].sum() * 100
).round(1)

print("=== SEGMENT SIZING ===")
print(sizing[["customer_count", "customer_share_pct",
              "total_revenue",  "revenue_share_pct"]].to_string())


# ── 3. BEHAVIOUR PROFILE TABLE ────────────────────────────────────────────────
print("\n=== BEHAVIOUR PROFILE (means per segment) ===")
profile = df.groupby(SEGMENT_COL)[FEATURES].mean().round(2)
print(profile.to_string())


# ── 4. CHART A — REVENUE & CUSTOMER SHARE BY SEGMENT ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Segment sizing", fontsize=14, fontweight="500", y=1.01)

seg_order = sizing.sort_values("total_revenue", ascending=False).index

# Revenue bar
axes[0].barh(seg_order,
             sizing.loc[seg_order, "total_revenue"],
             color=[PALETTE.get(s, "#888780") for s in seg_order])
axes[0].set_title("Total revenue by segment")
axes[0].set_xlabel("Revenue ($)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"${x:,.0f}"))
axes[0].invert_yaxis()

# Customer count bar
axes[1].barh(seg_order,
             sizing.loc[seg_order, "customer_count"],
             color=[PALETTE.get(s, "#888780") for s in seg_order])
axes[1].set_title("Customer count by segment")
axes[1].set_xlabel("Customers")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("output/chart_segment_sizing.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 5. CHART B — RFM HEATMAP ─────────────────────────────────────────────────
heatmap_data = profile[["recency_days", "frequency", "monetary",
                         "avg_order_value", "orders_per_month"]].copy()

# Normalize each column 0–1 so different scales are comparable
heatmap_norm = (heatmap_data - heatmap_data.min()) / \
               (heatmap_data.max() - heatmap_data.min())

# Invert recency — lower days is better, so we flip for visual consistency
heatmap_norm["recency_days"] = 1 - heatmap_norm["recency_days"]
heatmap_norm = heatmap_norm.rename(columns={"recency_days": "recency (inverted)"})

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(heatmap_norm,
            annot=heatmap_data.round(1),  # show raw values in cells
            fmt="g",
            cmap="YlGn",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Normalised score (0–1)"})

ax.set_title("Segment behaviour heatmap (normalised)", fontsize=13)
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("output/chart_rfm_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 6. CHART C — SCATTER: RECENCY vs MONETARY, SIZED BY FREQUENCY ────────────
fig, ax = plt.subplots(figsize=(10, 6))

for seg, grp in df.groupby(SEGMENT_COL):
    ax.scatter(
        grp["recency_days"],
        grp["monetary"],
        s=grp["frequency"] * 10,   # bubble size = order frequency
        alpha=0.45,
        label=seg,
        color=PALETTE.get(seg, "#888780"),
        edgecolors="none"
    )

ax.set_xlabel("Recency (days since last order) — lower is better")
ax.set_ylabel("Total spend ($)")
ax.set_title("Customer map — recency vs spend (bubble = frequency)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(title="Segment", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("output/chart_customer_map.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 7. CHART D — BOX PLOT: MONETARY DISTRIBUTION PER SEGMENT ─────────────────
fig, ax = plt.subplots(figsize=(11, 5))

seg_order_rev = sizing.sort_values("avg_revenue", ascending=False).index.tolist()
data_by_seg   = [df[df[SEGMENT_COL] == s]["monetary"].values for s in seg_order_rev]
colors_list   = [PALETTE.get(s, "#888780") for s in seg_order_rev]

bp = ax.boxplot(data_by_seg, labels=seg_order_rev, patch_artist=True,
                medianprops=dict(color="white", linewidth=2))

for patch, color in zip(bp["boxes"], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title("Revenue distribution per segment")
ax.set_ylabel("Total spend ($)")
ax.set_xlabel("")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("output/chart_revenue_distribution.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 8. BUSINESS RECOMMENDATIONS ───────────────────────────────────────────────
RECOMMENDATIONS = {
    "Champions": {
        "action":   "Reward & retain",
        "tactics":  [
            "Offer early access to new products",
            "Invite to loyalty / VIP programme",
            "Ask for reviews and referrals",
        ],
        "priority": "High"
    },
    "Loyal customers": {
        "action":   "Upsell & deepen",
        "tactics":  [
            "Cross-sell complementary products",
            "Offer subscription or bundle deals",
            "Personalise communications by purchase history",
        ],
        "priority": "High"
    },
    "At-risk customers": {
        "action":   "Re-engage urgently",
        "tactics":  [
            "Send win-back email with time-limited discount",
            "Survey to understand drop-off reason",
            "Highlight new arrivals since last purchase",
        ],
        "priority": "Critical"
    },
    "Lost / dormant": {
        "action":   "Low-cost reactivation",
        "tactics":  [
            "Last-chance win-back campaign",
            "Suppress from expensive paid channels",
            "Sunset if no response after 2 attempts",
        ],
        "priority": "Low"
    },
    "New customers": {
        "action":   "Onboard & convert",
        "tactics":  [
            "Send welcome series with product education",
            "Offer second-purchase incentive",
            "Guide toward most popular categories",
        ],
        "priority": "High"
    },
    "Frequent low-spenders": {
        "action":   "Increase basket size",
        "tactics":  [
            "Show minimum spend threshold for free shipping",
            "Bundle offers and add-ons at checkout",
            "Promote premium product lines",
        ],
        "priority": "Medium"
    },
    "Potential loyalists": {
        "action":   "Nurture toward loyalty",
        "tactics":  [
            "Enrol in loyalty points programme",
            "Send personalised recommendations",
            "Offer mid-tier membership benefits",
        ],
        "priority": "Medium"
    },
}

print("\n=== BUSINESS RECOMMENDATIONS ===")
for seg, rec in RECOMMENDATIONS.items():
    if seg in SEGMENTS:
        print(f"\n[{rec['priority']}] {seg} — {rec['action']}")
        for t in rec["tactics"]:
            print(f"   • {t}")


# ── 9. EXPORT ──────────────────────────────────────────────────────────────────
# Build flat recommendations table for Excel
rec_rows = []
for seg, rec in RECOMMENDATIONS.items():
    if seg in SEGMENTS:
        count   = int(sizing.loc[seg, "customer_count"]) if seg in sizing.index else 0
        revenue = float(sizing.loc[seg, "total_revenue"]) if seg in sizing.index else 0
        for tactic in rec["tactics"]:
            rec_rows.append({
                "segment":        seg,
                "priority":       rec["priority"],
                "action":         rec["action"],
                "tactic":         tactic,
                "customer_count": count,
                "total_revenue":  revenue,
            })

rec_df = pd.DataFrame(rec_rows)

with pd.ExcelWriter("output/segment_profiles.xlsx", engine="openpyxl") as writer:
    sizing.to_excel(writer,                   sheet_name="Segment Sizing")
    profile.to_excel(writer,                  sheet_name="Behaviour Profile")
    rec_df.to_excel(writer, index=False,      sheet_name="Recommendations")
    df.to_excel(writer,     index=False,      sheet_name="All Customers")

print("\nPhase 4 complete. Files saved to /output/")