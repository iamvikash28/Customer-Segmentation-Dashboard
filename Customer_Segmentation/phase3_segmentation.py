import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD FEATURE MATRIX FROM PHASE 2 ──────────────────────────────────────
df = pd.read_csv("output/feature_matrix.csv")

# Keep only customers who have made at least one purchase
df = df[df["frequency"] > 0].copy()
df = df.dropna(subset=["recency_days", "frequency", "monetary"])

print(f"Customers for clustering: {len(df)}")


# ── 2. SELECT FEATURES FOR CLUSTERING ────────────────────────────────────────
# Core RFM + extended features
FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "orders_per_month",
    "lifespan_days"
]

# Ensure all clustering features exist, fill missing with 0
for col in FEATURES:
    if col not in df.columns:
        df[col] = 0

X = df[FEATURES].copy()


# ── 3. SCALE FEATURES ────────────────────────────────────────────────────────
# K-Means is distance-based — unscaled features will bias toward high-magnitude
# columns like monetary. StandardScaler brings all features to mean=0, std=1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features scaled. Mean ~0, Std ~1 per column.")


# ── 4. FIND OPTIMAL K — ELBOW METHOD ─────────────────────────────────────────
inertias   = []
sil_scores = []
K_RANGE    = range(2, 11)

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"  K={k}  inertia={km.inertia_:.0f}  silhouette={sil_scores[-1]:.3f}")

# Plot elbow + silhouette side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(list(K_RANGE), inertias, "o-", color="#378ADD", linewidth=2)
ax1.set_title("Elbow method — inertia vs K")
ax1.set_xlabel("Number of clusters (K)")
ax1.set_ylabel("Inertia")
ax1.grid(True, alpha=0.3)

ax2.plot(list(K_RANGE), sil_scores, "o-", color="#1D9E75", linewidth=2)
ax2.set_title("Silhouette score vs K")
ax2.set_xlabel("Number of clusters (K)")
ax2.set_ylabel("Silhouette score (higher = better)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nElbow chart saved → output/elbow_silhouette.png")


# ── 5. CHOOSE K AND FIT FINAL MODEL ──────────────────────────────────────────
# Set K to the elbow point — typically where inertia drop flattens.
# Common result for customer segmentation is K=4 or K=5.
# Adjust after reviewing the chart above.
OPTIMAL_K = 4   # <-- change this after reviewing your elbow chart

kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df["cluster_id"] = kmeans.fit_predict(X_scaled)

print(f"\nFinal model: K={OPTIMAL_K}")
print(f"Silhouette score: {silhouette_score(X_scaled, df['cluster_id']):.3f}")
print(f"\nCluster sizes:\n{df['cluster_id'].value_counts().sort_index()}")


# ── 6. PROFILE EACH CLUSTER ───────────────────────────────────────────────────
cluster_profile = df.groupby("cluster_id")[FEATURES].mean().round(2)
cluster_profile["customer_count"] = df.groupby("cluster_id")["customer_id"].count()

print("\n=== CLUSTER PROFILES ===")
print(cluster_profile.T)


# ── 7. ASSIGN BUSINESS LABELS ────────────────────────────────────────────────
# After reviewing cluster_profile, assign intuitive names.
# The mapping below is a starting template — update based on YOUR data.
#
# How to read profiles:
#   Low recency_days + high frequency + high monetary  → Champions
#   High recency_days + low frequency + low monetary   → Lost / Dormant
#   Mid recency + high frequency + low monetary        → Frequent low-spenders
#   Low recency + low frequency + high monetary        → High-value new buyers

CLUSTER_LABELS = {
    0: "Champions",           # high RFM across the board
    1: "Loyal customers",     # solid frequency, moderate monetary
    2: "At-risk customers",   # used to buy but recency is high
    3: "Lost / dormant",      # low on all three
}
# If OPTIMAL_K=5, add a fifth label e.g. 4: "New customers"

df["segment_label"] = df["cluster_id"].map(CLUSTER_LABELS)

print("\n=== SEGMENT DISTRIBUTION ===")
print(df["segment_label"].value_counts())


# ── 8. SILHOUETTE PLOT PER CLUSTER ───────────────────────────────────────────
# Shows how well each customer fits its cluster (score near 1 = tight fit)
sil_vals   = silhouette_samples(X_scaled, df["cluster_id"])
fig, ax    = plt.subplots(figsize=(8, 5))
y_lower    = 10

for i in range(OPTIMAL_K):
    ith_vals = np.sort(sil_vals[df["cluster_id"] == i])
    size_i   = ith_vals.shape[0]
    y_upper  = y_lower + size_i
    color    = cm.nipy_spectral(float(i) / OPTIMAL_K)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals,
                     facecolor=color, edgecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
    y_lower  = y_upper + 10

ax.axvline(x=silhouette_score(X_scaled, df["cluster_id"]),
           color="red", linestyle="--", label="Avg silhouette")
ax.set_title("Silhouette plot by cluster")
ax.set_xlabel("Silhouette coefficient")
ax.set_ylabel("Cluster")
ax.legend()
plt.tight_layout()
plt.savefig("output/silhouette_plot.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 9. EXPORT LABELED SEGMENTS ────────────────────────────────────────────────
df.to_csv("output/segments_labeled.csv", index=False)

with pd.ExcelWriter("output/segmentation_results.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="All Customers", index=False)
    cluster_profile.to_excel(writer, sheet_name="Cluster Profiles")
    df["segment_label"].value_counts().to_frame("count").to_excel(
        writer, sheet_name="Segment Counts")

print("\nPhase 3 complete. Files saved to /output/")