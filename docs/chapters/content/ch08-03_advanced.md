# Ch 8: Unsupervised Learning - Advanced

**Track**: Practitioner | [Try code in Playground](../../playground.md) | [Back to chapter overview](../chapter-08.md)


!!! tip "Read online or run locally"
    You can read this content here on the web. To run the code interactively,
    either use the [Playground](../../playground.md) or clone the repo and open
    `chapters/chapter-08-unsupervised-learning/notebooks/03_advanced.ipynb` in Jupyter.

---

# Chapter 8: Unsupervised Learning
## Notebook 03 - Advanced: Dimensionality Reduction & Capstone

Reduce high-dimensional data for visualization and modeling, detect anomalies, and build a complete customer segmentation system.

**What you'll learn:**
- Principal Component Analysis (PCA) from scratch
- t-SNE for 2D visualization
- Anomaly detection with Isolation Forest
- Customer segmentation capstone project

**Time estimate:** 3 hours

---

## 1. PCA Theory

### The Core Idea

PCA is a **linear** dimensionality-reduction technique that finds the directions (called **principal components**) along which the data varies the most.

Imagine a cloud of 3-D points shaped like a flat pancake. Two axes capture almost all of the spread; the third adds very little information. PCA discovers those two dominant axes automatically.

### Algorithm Steps

1. **Center the data** — subtract the mean of each feature so that the cloud is centered at the origin.
2. **Compute the covariance matrix** — a \(d \times d\) matrix (where \(d\) is the number of features) that captures pairwise linear relationships.
3. **Eigendecomposition** — find the eigenvectors and eigenvalues of the covariance matrix. Each eigenvector is a principal component direction; its eigenvalue tells us how much variance that direction explains.
4. **Sort & select** — rank components by eigenvalue (descending) and keep the top \(k\) to reduce dimensionality from \(d\) to \(k\).
5. **Project** — multiply the centered data by the selected eigenvectors to obtain the lower-dimensional representation.

### Variance Explained Ratio

The variance explained ratio for component \(i\) is \(\lambda_i / \sum_{j=1}^{d} \lambda_j\), where \(\lambda_i\) is the \(i\)-th eigenvalue. The **cumulative** variance explained tells us how much total information is retained when we keep the first \(k\) components.

---

## 2. PCA From Scratch

We implement PCA using only NumPy and apply it to the classic **Iris** dataset (4 features → 2 components).

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

np.random.seed(42)

# Load the Iris dataset (4 features, 150 samples, 3 classes)
iris = load_iris()
X = iris.data          # shape (150, 4)
y = iris.target        # 0, 1, 2
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Classes:  {list(target_names)}")
```

```python
def pca_from_scratch(X, n_components=2):
    """Implement PCA using NumPy."""
    # Step 1: Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Step 2: Covariance matrix (features × features)
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort by eigenvalue descending
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Variance explained ratio
    variance_ratio = eigenvalues / eigenvalues.sum()

    # Step 5: Project onto top-k components
    W = eigenvectors[:, :n_components]
    X_projected = X_centered @ W

    return X_projected, eigenvalues, variance_ratio, W


X_pca_scratch, eigenvalues, var_ratio, components = pca_from_scratch(X, n_components=2)

print("Eigenvalues:", np.round(eigenvalues, 4))
print("Variance explained ratio:", np.round(var_ratio, 4))
print(f"Total variance retained (2 components): {var_ratio[:2].sum():.2%}")
```

```python
# Variance Explained Bar + Cumulative Line
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: bar chart of individual variance ratios
axes[0].bar(range(1, len(var_ratio) + 1), var_ratio, color="steelblue", edgecolor="black")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained Ratio")
axes[0].set_title("Variance Explained by Each Component")
axes[0].set_xticks(range(1, len(var_ratio) + 1))

# Right: cumulative variance explained
cumulative = np.cumsum(var_ratio)
axes[1].plot(range(1, len(cumulative) + 1), cumulative, "o-", color="darkorange", linewidth=2)
axes[1].axhline(y=0.95, color="red", linestyle="--", label="95% threshold")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance Explained")
axes[1].set_title("Cumulative Variance Explained")
axes[1].set_xticks(range(1, len(cumulative) + 1))
axes[1].legend()

plt.tight_layout()
plt.show()
```

```python
# 2-D scatter plot of the scratch PCA projection
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

plt.figure(figsize=(8, 6))
for i, name in enumerate(target_names):
    mask = y == i
    plt.scatter(X_pca_scratch[mask, 0], X_pca_scratch[mask, 1],
                label=name, alpha=0.7, edgecolors="k", linewidth=0.5,
                color=colors[i], s=60)
plt.xlabel(f"PC 1 ({var_ratio[0]:.1%} variance)")
plt.ylabel(f"PC 2 ({var_ratio[1]:.1%} variance)")
plt.title("PCA From Scratch — Iris Dataset (2-D Projection)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 3. PCA with Scikit-learn

We verify our scratch implementation against the well-optimized `sklearn.decomposition.PCA`.

```python
from sklearn.decomposition import PCA

pca_sk = PCA(n_components=4)  # keep all 4 to inspect variance
X_pca_sk_full = pca_sk.fit_transform(X)

print("Sklearn variance explained ratio:", np.round(pca_sk.explained_variance_ratio_, 4))
print("Scratch variance explained ratio: ", np.round(var_ratio, 4))
print()
print("Cumulative (sklearn):", np.round(np.cumsum(pca_sk.explained_variance_ratio_), 4))
```

```python
X_pca_sk = X_pca_sk_full[:, :2]  # first 2 components

# Sign of eigenvectors can flip — align for visual comparison
for col in range(2):
    if np.corrcoef(X_pca_scratch[:, col], X_pca_sk[:, col])[0, 1] < 0:
        X_pca_scratch[:, col] *= -1

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

for ax, data, title in zip(axes,
                            [X_pca_scratch, X_pca_sk],
                            ["PCA (from scratch)", "PCA (scikit-learn)"]):
    for i, name in enumerate(target_names):
        mask = y == i
        ax.scatter(data[mask, 0], data[mask, 1], label=name,
                   alpha=0.7, edgecolors="k", linewidth=0.5,
                   color=colors[i], s=60)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle("Scratch vs Scikit-learn PCA — Identical Results", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

The two plots are virtually identical (eigenvector signs may differ, which is cosmetic). This confirms our from-scratch implementation is correct.

---

## 4. t-SNE

### What is t-SNE?

**t-distributed Stochastic Neighbor Embedding (t-SNE)** is a non-linear dimensionality-reduction technique designed specifically for **visualization**.

Key properties:
- Preserves **local structure**: points that are close in high-dimensional space stay close in the 2-D embedding.
- Does **not** preserve global distances — clusters may move relative to each other between runs.
- Computationally expensive — not suitable as a preprocessing step in machine-learning pipelines.
- The **perplexity** parameter (roughly: how many neighbors each point considers) strongly influences the result. Typical range: 5–50.

**Rule of thumb:** Use PCA when you need a general-purpose reduction (for modeling, compression, noise removal). Use t-SNE when your sole goal is to *see* cluster structure in 2-D.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X)

print(f"t-SNE output shape: {X_tsne.shape}")
```

```python
# Side-by-side: PCA vs t-SNE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, data, title in zip(axes,
                            [X_pca_sk, X_tsne],
                            ["PCA (linear)", "t-SNE (non-linear)"]):
    for i, name in enumerate(target_names):
        mask = y == i
        ax.scatter(data[mask, 0], data[mask, 1], label=name,
                   alpha=0.7, edgecolors="k", linewidth=0.5,
                   color=colors[i], s=60)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle("PCA vs t-SNE — Iris Dataset", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

```python
# Effect of perplexity on t-SNE
perplexities = [5, 15, 30, 50]
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for ax, perp in zip(axes, perplexities):
    embedding = TSNE(n_components=2, perplexity=perp,
                     random_state=42, n_iter=1000).fit_transform(X)
    for i, name in enumerate(target_names):
        mask = y == i
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   alpha=0.7, color=colors[i], s=40, edgecolors="k",
                   linewidth=0.3, label=name)
    ax.set_title(f"Perplexity = {perp}")
    ax.set_xticks([])
    ax.set_yticks([])

axes[0].legend(fontsize=8)
plt.suptitle("t-SNE: Impact of Perplexity", fontsize=14, y=1.04)
plt.tight_layout()
plt.show()
```

**Observations on perplexity:**
- Low perplexity (5): focuses on very local neighbors — clusters may fragment.
- High perplexity (50): considers more neighbors — clusters become rounder and more global structure is visible, but fine local detail may blur.
- There is no single "correct" perplexity; try several and look for consistent patterns.

---

## 5. Anomaly Detection

### Why Unsupervised Anomaly Detection?

In many real-world scenarios, labeled anomalies are scarce or non-existent:

| Domain | Normal | Anomaly |
|--------|--------|---------|
| Banking | Legitimate transactions | Fraud |
| Manufacturing | Good products | Defects |
| Cybersecurity | Regular traffic | Intrusions |

Unsupervised methods learn the distribution of *normal* data and flag anything that doesn't fit.

### Approach 1 — Z-Score

Flag a point as anomalous if any feature has a Z-score \(|z| > \tau\) (e.g., \(\tau = 3\)). Simple, but assumes Gaussian features and works only for univariate or low-dimensional data.

### Approach 2 — Isolation Forest

The **Isolation Forest** algorithm isolates observations by randomly selecting a feature and a split value. Anomalies are easier to isolate (fewer splits needed), so they have shorter average path lengths in the trees.

Advantages:
- Works well in high dimensions
- No distribution assumptions
- Linear time complexity

```python
from sklearn.ensemble import IsolationForest
from scipy import stats

np.random.seed(42)

# Generate normal data: 2 clusters
normal_a = np.random.randn(150, 2) * 0.8 + np.array([2, 2])
normal_b = np.random.randn(150, 2) * 0.8 + np.array([-2, -2])
normal_data = np.vstack([normal_a, normal_b])

# Inject 20 anomalies scattered far from the clusters
anomalies = np.random.uniform(low=-6, high=6, size=(20, 2))

X_anom = np.vstack([normal_data, anomalies])
labels_true = np.array([0] * len(normal_data) + [1] * len(anomalies))  # 0=normal, 1=anomaly

print(f"Total points: {len(X_anom)}  (normal: {len(normal_data)}, anomalies: {len(anomalies)})")
```

```python
# Z-Score method
z_scores = np.abs(stats.zscore(X_anom))
z_threshold = 3.0
z_anomaly_mask = (z_scores > z_threshold).any(axis=1)

print(f"Z-Score method detected {z_anomaly_mask.sum()} anomalies (threshold={z_threshold})")
```

```python
# Isolation Forest
iso_forest = IsolationForest(n_estimators=200, contamination=0.06,
                            random_state=42)
iso_preds = iso_forest.fit_predict(X_anom)  # 1 = normal, -1 = anomaly
iso_anomaly_mask = iso_preds == -1

print(f"Isolation Forest detected {iso_anomaly_mask.sum()} anomalies")
```

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Ground truth
axes[0].scatter(X_anom[labels_true == 0, 0], X_anom[labels_true == 0, 1],
                c="steelblue", s=30, alpha=0.6, label="Normal")
axes[0].scatter(X_anom[labels_true == 1, 0], X_anom[labels_true == 1, 1],
                c="red", s=80, marker="X", label="True Anomaly")
axes[0].set_title("Ground Truth")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Z-Score
axes[1].scatter(X_anom[~z_anomaly_mask, 0], X_anom[~z_anomaly_mask, 1],
                c="steelblue", s=30, alpha=0.6, label="Normal")
axes[1].scatter(X_anom[z_anomaly_mask, 0], X_anom[z_anomaly_mask, 1],
                c="red", s=80, marker="X", label="Detected Anomaly")
axes[1].set_title(f"Z-Score (threshold={z_threshold})")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Isolation Forest
axes[2].scatter(X_anom[~iso_anomaly_mask, 0], X_anom[~iso_anomaly_mask, 1],
                c="steelblue", s=30, alpha=0.6, label="Normal")
axes[2].scatter(X_anom[iso_anomaly_mask, 0], X_anom[iso_anomaly_mask, 1],
                c="red", s=80, marker="X", label="Detected Anomaly")
axes[2].set_title("Isolation Forest")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.suptitle("Anomaly Detection Comparison", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

**Key takeaway:** The Isolation Forest typically outperforms the Z-Score method, especially when the data is multi-modal or the anomalies are not simply extreme values along a single axis.

---

## 6. Capstone — Customer Segmentation

We build a complete customer-segmentation pipeline:

1. Generate & save a synthetic customer dataset
2. Feature scaling
3. Dimensionality reduction with PCA
4. Elbow method to choose optimal \(K\)
5. K-Means clustering
6. Segment profiling & visualization
7. Business recommendations

### 6.1 Generate Synthetic Customer Data

We create five features that mimic a retail scenario:

| Feature | Description |
|---------|-------------|
| `age` | Customer age (18–70) |
| `income` | Annual income in $k (15–150) |
| `spending_score` | In-store spending score (1–100) |
| `visits` | Monthly store visits (0–30) |
| `online_ratio` | Fraction of purchases made online (0–1) |

```python
import pandas as pd
import os

np.random.seed(42)
n_customers = 500

# Segment 1: Young, moderate income, high online, high spending
seg1 = {
    "age": np.random.normal(25, 4, 130).clip(18, 40),
    "income": np.random.normal(45, 12, 130).clip(15, 80),
    "spending_score": np.random.normal(75, 10, 130).clip(1, 100),
    "visits": np.random.normal(8, 3, 130).clip(0, 30),
    "online_ratio": np.random.normal(0.75, 0.1, 130).clip(0, 1),
}

# Segment 2: Middle-aged, high income, balanced channel, moderate spending
seg2 = {
    "age": np.random.normal(42, 6, 150).clip(28, 60),
    "income": np.random.normal(95, 18, 150).clip(50, 150),
    "spending_score": np.random.normal(55, 12, 150).clip(1, 100),
    "visits": np.random.normal(15, 5, 150).clip(0, 30),
    "online_ratio": np.random.normal(0.45, 0.15, 150).clip(0, 1),
}

# Segment 3: Older, lower income, low online, low spending
seg3 = {
    "age": np.random.normal(58, 7, 120).clip(40, 70),
    "income": np.random.normal(35, 10, 120).clip(15, 70),
    "spending_score": np.random.normal(25, 10, 120).clip(1, 100),
    "visits": np.random.normal(20, 5, 120).clip(0, 30),
    "online_ratio": np.random.normal(0.15, 0.08, 120).clip(0, 1),
}

# Segment 4: Mixed ages, very high income, high spending, moderate visits
seg4 = {
    "age": np.random.normal(38, 10, 100).clip(18, 70),
    "income": np.random.normal(120, 15, 100).clip(80, 150),
    "spending_score": np.random.normal(85, 8, 100).clip(1, 100),
    "visits": np.random.normal(12, 4, 100).clip(0, 30),
    "online_ratio": np.random.normal(0.55, 0.15, 100).clip(0, 1),
}

frames = []
for seg in [seg1, seg2, seg3, seg4]:
    frames.append(pd.DataFrame(seg))

df_customers = pd.concat(frames, ignore_index=True)
df_customers = df_customers.sample(frac=1, random_state=42).reset_index(drop=True)

df_customers["age"] = df_customers["age"].round(0).astype(int)
df_customers["income"] = df_customers["income"].round(1)
df_customers["spending_score"] = df_customers["spending_score"].round(0).astype(int)
df_customers["visits"] = df_customers["visits"].round(0).astype(int)
df_customers["online_ratio"] = df_customers["online_ratio"].round(2)

# Save to CSV (run from chapter folder: chapters/chapter-08-unsupervised-learning/)
dataset_dir = "datasets"
os.makedirs(dataset_dir, exist_ok=True)
csv_path = os.path.join(dataset_dir, "customers.csv")
df_customers.to_csv(csv_path, index=False)
print(f"Saved {len(df_customers)} rows to {csv_path}")
df_customers.head(10)
```

### 6.2 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

feature_cols = ["age", "income", "spending_score", "visits", "online_ratio"]
X_cust = df_customers[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cust)

print("Scaled means (≈0):", np.round(X_scaled.mean(axis=0), 4))
print("Scaled stds  (≈1):", np.round(X_scaled.std(axis=0), 4))
```

### 6.3 PCA for Dimensionality Reduction

```python
pca_cust = PCA(n_components=5)
X_pca_cust = pca_cust.fit_transform(X_scaled)

cum_var = np.cumsum(pca_cust.explained_variance_ratio_)

plt.figure(figsize=(7, 4))
plt.bar(range(1, 6), pca_cust.explained_variance_ratio_,
        color="steelblue", edgecolor="black", alpha=0.7, label="Individual")
plt.step(range(1, 6), cum_var, where="mid", color="darkorange",
         linewidth=2, label="Cumulative")
plt.axhline(0.90, color="red", linestyle="--", alpha=0.7, label="90% threshold")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.title("Customer Data — PCA Variance Explained")
plt.xticks(range(1, 6))
plt.legend()
plt.tight_layout()
plt.show()

n_keep = np.argmax(cum_var >= 0.90) + 1
print(f"\nComponents needed for ≥90% variance: {n_keep}")
print(f"Using first 2 components for visualization ({cum_var[1]:.1%} variance).")
```

### 6.4 K-Means — Elbow Method

```python
from sklearn.cluster import KMeans

K_range = range(2, 11)
inertias = []

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(list(K_range), inertias, "o-", linewidth=2, color="steelblue")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("Elbow Method for Optimal K")
plt.xticks(list(K_range))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Look for the 'elbow' — the point where adding more clusters yields")
print("diminishing returns. Here K=4 appears to be a good choice.")
```

### 6.5 Fit K-Means with Optimal K

```python
optimal_k = 4
km_final = KMeans(n_clusters=optimal_k, n_init=20, random_state=42)
cluster_labels = km_final.fit_predict(X_scaled)

df_customers["cluster"] = cluster_labels
print(f"Cluster distribution:\n{df_customers['cluster'].value_counts().sort_index()}")
```

### 6.6 Segment Profiling

```python
segment_profile = df_customers.groupby("cluster")[feature_cols].mean().round(2)
segment_profile["count"] = df_customers.groupby("cluster").size()
print("=== Segment Profiles ===")
segment_profile
```

```python
# Radar / parallel-coordinates style comparison
fig, axes = plt.subplots(1, len(feature_cols), figsize=(18, 4), sharey=True)
cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for idx, feat in enumerate(feature_cols):
    means = df_customers.groupby("cluster")[feat].mean()
    axes[idx].bar(means.index, means.values,
                  color=cluster_colors[:optimal_k], edgecolor="black")
    axes[idx].set_title(feat, fontsize=11)
    axes[idx].set_xlabel("Cluster")
    axes[idx].set_xticks(range(optimal_k))

axes[0].set_ylabel("Mean Value")
plt.suptitle("Feature Means by Cluster", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

### 6.7 Visualize Segments in 2-D (PCA Projection)

```python
X_vis = X_pca_cust[:, :2]
centroids_scaled = km_final.cluster_centers_
centroids_2d = pca_cust.transform(centroids_scaled)[:, :2]  # project centroids

plt.figure(figsize=(9, 7))
for c in range(optimal_k):
    mask = cluster_labels == c
    plt.scatter(X_vis[mask, 0], X_vis[mask, 1], s=40, alpha=0.6,
                color=cluster_colors[c], edgecolors="k", linewidth=0.3,
                label=f"Segment {c}")

plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=250, c="black",
            marker="*", zorder=5, label="Centroids")

plt.xlabel(f"PC 1 ({pca_cust.explained_variance_ratio_[0]:.1%} var)")
plt.ylabel(f"PC 2 ({pca_cust.explained_variance_ratio_[1]:.1%} var)")
plt.title("Customer Segments — PCA 2-D Projection")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### 6.8 Business Recommendations

```python
recommendations = {
    0: {
        "label": "Budget Traditionalists",
        "description": "Older customers with low income and spending, who shop mostly in-store.",
        "actions": [
            "Offer loyalty discounts and in-store promotions",
            "Simplify the in-store experience",
            "Provide personalized coupons at checkout",
        ],
    },
    1: {
        "label": "Young Digital Shoppers",
        "description": "Young customers with moderate income but high online engagement and spending.",
        "actions": [
            "Invest in mobile app features and social media marketing",
            "Offer free shipping and digital-only deals",
            "Launch a referral program to leverage their network",
        ],
    },
    2: {
        "label": "Premium High-Spenders",
        "description": "High income, high spending score — the most valuable segment.",
        "actions": [
            "Create a VIP/premium loyalty tier",
            "Offer early access to new products",
            "Assign dedicated account managers for retention",
        ],
    },
    3: {
        "label": "Established Moderates",
        "description": "Middle-aged, higher income, moderate spending, balanced channel use.",
        "actions": [
            "Cross-sell higher-margin products",
            "Provide omni-channel convenience (buy online, pick up in store)",
            "Target with email campaigns for seasonal offers",
        ],
    },
}

for seg_id, info in recommendations.items():
    count = (cluster_labels == seg_id).sum()
    print(f"\n{'='*60}")
    print(f"Segment {seg_id}: {info['label']}  (n={count})")
    print(f"{'='*60}")
    print(f"  {info['description']}")
    print("  Recommended actions:")
    for action in info["actions"]:
        print(f"    • {action}")
```

---

## 7. Summary

### What We Covered in This Notebook

| Topic | Key Idea |
|-------|----------|
| **PCA** | Linear projection onto directions of maximum variance |
| **t-SNE** | Non-linear embedding that preserves local neighborhoods — for visualization only |
| **Z-Score Anomaly Detection** | Simple threshold on standardized values |
| **Isolation Forest** | Tree-based anomaly detector — fast, distribution-free |
| **Customer Segmentation** | End-to-end pipeline: scale → PCA → K-Means → profile → recommend |

### Chapter 8 Recap

Across the three notebooks you have:

1. **Notebook 01 (Introduction):** Learned K-Means, hierarchical clustering, and evaluation metrics.
2. **Notebook 02 (Intermediate):** Explored DBSCAN, Gaussian Mixture Models, and silhouette analysis.
3. **Notebook 03 (Advanced — this one):** Mastered PCA, t-SNE, anomaly detection, and built a full capstone project.

### What's Next

In **Chapter 9: Deep Learning** we'll move from classical ML to neural networks — starting with perceptrons, backpropagation, and building your first deep network with PyTorch/Keras.

---
*Generated by Berta AI | Created by Luigi Pascal Rondanini*
