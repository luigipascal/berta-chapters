# Ch 8: Unsupervised Learning - Introduction

**Track**: Practitioner | [Try code in Playground](../../playground.md) | [Back to chapter overview](../chapter-08.md)


!!! tip "Read online or run locally"
    You can read this content here on the web. To run the code interactively,
    either use the [Playground](../../playground.md) or clone the repo and open
    `chapters/chapter-08-unsupervised-learning/notebooks/01_introduction.ipynb` in Jupyter.

---

# Chapter 8: Unsupervised Learning
## Notebook 01 - Introduction: Clustering Basics

Unsupervised learning finds hidden patterns in data without labels. We start with the most fundamental algorithm: **K-Means clustering**.

**What you'll learn:**
- The difference between supervised and unsupervised learning
- K-Means clustering from scratch using NumPy
- Evaluating clusters with inertia and silhouette score
- The elbow method for choosing K
- Scikit-learn's KMeans interface

**Time estimate:** 2.5 hours

---

## 1. Supervised vs Unsupervised Learning

In **supervised learning**, every training example comes with a label — the "right answer" — and the model learns a mapping from inputs to outputs. Classification and regression are the classic examples.

In **unsupervised learning**, there are **no labels at all**. The algorithm must discover structure in the data on its own. Common tasks include clustering (group similar points), dimensionality reduction (compress features), and anomaly detection (find unusual observations).

This notebook focuses on **clustering** — specifically the **K-Means** algorithm. Let's start by generating some data and seeing what it looks like *without* labels. The left plot shows raw data (all same color); the right reveals the true clusters we want the algorithm to recover on its own.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(42)

X, y_true = make_blobs(
    n_samples=200, centers=3, cluster_std=0.9, random_state=42
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(X[:, 0], X[:, 1], c="steelblue", edgecolors="k", s=50, alpha=0.7)
axes[0].set_title("What we observe (no labels)", fontsize=14)
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

colors = ["#e74c3c", "#2ecc71", "#3498db"]
for k in range(3):
    mask = y_true == k
    axes[1].scatter(X[mask, 0], X[mask, 1], c=colors[k],
                    edgecolors="k", s=50, alpha=0.7, label=f"Cluster {k}")
axes[1].set_title("True clusters (hidden from algorithm)", fontsize=14)
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## 2. K-Means Algorithm

K-Means is an iterative algorithm that partitions *n* data points into *K* clusters. It works in three repeating steps:

**Step 1 — Initialize:** Pick *K* points as initial **centroids** (cluster centers). The simplest approach is to choose *K* data points at random.

**Step 2 — Assign:** For every data point, compute the Euclidean distance to each centroid and assign the point to the **nearest** centroid.

**Step 3 — Update:** Recompute each centroid as the **mean** of all points currently assigned to that cluster.

**Repeat** Steps 2 and 3 until the assignments no longer change (or a maximum number of iterations is reached).

Let's implement K-Means from scratch using only NumPy:

```python
class KMeansScratch:
    """Minimal K-Means implementation using NumPy."""

    def __init__(self, k=3, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.inertia_history = []
        self.centroid_history = []
        self.label_history = []

    def _euclidean_distances(self, X, centroids):
        """Compute distance from every point to every centroid."""
        return np.sqrt(((X[:, np.newaxis] - centroids[np.newaxis]) ** 2).sum(axis=2))

    def _compute_inertia(self, X, labels, centroids):
        return sum(
            np.sum((X[labels == k] - centroids[k]) ** 2)
            for k in range(self.k)
        )

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        # Step 1: random initialization
        idx = rng.choice(n_samples, self.k, replace=False)
        self.centroids = X[idx].copy()

        self.inertia_history = []
        self.centroid_history = [self.centroids.copy()]
        self.label_history = []

        for _ in range(self.max_iters):
            # Step 2: assign
            distances = self._euclidean_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)
            self.label_history.append(labels.copy())

            # Step 3: update centroids
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k)
                else self.centroids[k]
                for k in range(self.k)
            ])

            inertia = self._compute_inertia(X, labels, new_centroids)
            self.inertia_history.append(inertia)
            self.centroid_history.append(new_centroids.copy())

            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids

        self.labels_ = labels
        self.inertia_ = self.inertia_history[-1]
        return self

    def predict(self, X):
        distances = self._euclidean_distances(X, self.centroids)
        return np.argmin(distances, axis=1)


km_scratch = KMeansScratch(k=3, random_state=42)
km_scratch.fit(X)

print(f"Converged in {len(km_scratch.inertia_history)} iterations")
print(f"Final inertia: {km_scratch.inertia_:.2f}")
print(f"Centroids:\n{km_scratch.centroids}")
```

Now let's plot the ground truth alongside our K-Means result:

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors_map = np.array(["#e74c3c", "#2ecc71", "#3498db"])

for k in range(3):
    mask = y_true == k
    axes[0].scatter(X[mask, 0], X[mask, 1], c=colors[k],
                    edgecolors="k", s=50, alpha=0.7, label=f"True {k}")
axes[0].set_title("Ground Truth", fontsize=14)
axes[0].legend()
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

axes[1].scatter(X[:, 0], X[:, 1], c=colors_map[km_scratch.labels_],
                edgecolors="k", s=50, alpha=0.7)
axes[1].scatter(km_scratch.centroids[:, 0], km_scratch.centroids[:, 1],
                c=colors, marker="X", s=250, edgecolors="k", linewidths=1.5,
                zorder=5, label="Centroids")
axes[1].set_title("K-Means (scratch) result", fontsize=14)
axes[1].legend()
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")

plt.tight_layout()
plt.show()
```

---

## 3. Step-by-Step Visualization

To build intuition for how the algorithm converges, let's watch the first four iterations unfold. Each subplot shows the cluster assignments and centroid positions at a particular iteration. Notice how the centroids migrate toward the cluster centers with each iteration.

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

colors_map = np.array(["#e74c3c", "#2ecc71", "#3498db"])

n_show = min(4, len(km_scratch.label_history))

for i in range(n_show):
    ax = axes[i]
    labels_i = km_scratch.label_history[i]
    centroids_i = km_scratch.centroid_history[i]
    centroids_next = km_scratch.centroid_history[i + 1]

    ax.scatter(X[:, 0], X[:, 1], c=colors_map[labels_i],
               edgecolors="k", s=40, alpha=0.6)

    ax.scatter(centroids_i[:, 0], centroids_i[:, 1],
               facecolors="none", edgecolors="k", marker="o",
               s=200, linewidths=2, label="Old centroid")

    ax.scatter(centroids_next[:, 0], centroids_next[:, 1],
               c=colors, marker="X", s=250, edgecolors="k",
               linewidths=1.5, zorder=5, label="New centroid")

    for k in range(3):
        ax.annotate("",
                    xy=centroids_next[k], xytext=centroids_i[k],
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    ax.set_title(f"Iteration {i + 1}  |  inertia = {km_scratch.inertia_history[i]:.1f}",
                 fontsize=12)
    if i == 0:
        ax.legend(fontsize=9, loc="upper left")

for j in range(n_show, 4):
    axes[j].axis("off")

plt.suptitle("K-Means — Iteration-by-Iteration", fontsize=15, y=1.01)
plt.tight_layout()
plt.show()
```

---

## 4. Evaluating Clusters

How do we know if K-Means did a good job? Two common metrics:

**Inertia (Within-Cluster Sum of Squares):** The sum of squared distances from each point to its centroid. Lower is better, but inertia *always* decreases as K increases — so it alone doesn't tell us the right K.

**Silhouette Score:** For each point, we compare the mean distance to others in the same cluster (*a*) vs. the mean distance to the nearest other cluster (*b*). The score is \( \frac{b - a}{\max(a, b)} \), ranging from −1 to +1. Higher is better; values near 0 indicate overlapping clusters.

```python
from sklearn.metrics import silhouette_score, silhouette_samples

sil_avg = silhouette_score(X, km_scratch.labels_)
sil_vals = silhouette_samples(X, km_scratch.labels_)

print(f"Inertia:              {km_scratch.inertia_:.2f}")
print(f"Silhouette (mean):    {sil_avg:.4f}")
print(f"Silhouette (min):     {sil_vals.min():.4f}")
print(f"Silhouette (max):     {sil_vals.max():.4f}")
```

A silhouette plot shows each cluster's distribution of silhouette coefficients. Healthy clusters extend well past the mean line; thin slivers or clusters barely crossing zero suggest poor separation.

```python
fig, ax = plt.subplots(figsize=(8, 5))

y_lower = 10
colors_sil = ["#e74c3c", "#2ecc71", "#3498db"]

for k in range(3):
    cluster_sil = np.sort(sil_vals[km_scratch.labels_ == k])
    cluster_size = cluster_sil.shape[0]
    y_upper = y_lower + cluster_size

    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                     facecolor=colors_sil[k], edgecolor=colors_sil[k], alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * cluster_size, f"Cluster {k}", fontsize=11,
            fontweight="bold", va="center")
    y_lower = y_upper + 10

ax.axvline(x=sil_avg, color="k", linestyle="--", linewidth=1.5,
           label=f"Mean silhouette = {sil_avg:.3f}")
ax.set_xlabel("Silhouette coefficient", fontsize=12)
ax.set_ylabel("Points (sorted within cluster)", fontsize=12)
ax.set_title("Silhouette Plot — K-Means (K=3)", fontsize=14)
ax.legend(fontsize=11)
ax.set_yticks([])
plt.tight_layout()
plt.show()
```

---

## 5. The Elbow Method

Since we must specify *K* before running K-Means, how do we pick a good value?

**The Elbow Method:**
1. Run K-Means for \( K = 1, 2, \ldots, K_{\max} \).
2. Plot inertia vs K.
3. Look for the **"elbow"** — the point where inertia stops decreasing sharply and begins to level off.

We can also plot silhouette score vs K; the best K often maximizes silhouette. Both plots together give a clearer picture.

```python
K_range = range(1, 11)
inertias = []
silhouettes = []

for k in K_range:
    km = KMeansScratch(k=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
    if k >= 2:
        silhouettes.append(silhouette_score(X, km.labels_))
    else:
        silhouettes.append(np.nan)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, "o-", color="#2c3e50", linewidth=2, markersize=8)
axes[0].set_xlabel("Number of clusters (K)", fontsize=12)
axes[0].set_ylabel("Inertia", fontsize=12)
axes[0].set_title("Elbow Method", fontsize=14)
axes[0].axvline(x=3, color="#e74c3c", linestyle="--", alpha=0.7, label="K = 3 (elbow)")
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

sil_values = [s for s in silhouettes if not np.isnan(s)]
sil_ks = list(range(2, 11))
axes[1].plot(sil_ks, sil_values, "s-", color="#27ae60", linewidth=2, markersize=8)
axes[1].set_xlabel("Number of clusters (K)", fontsize=12)
axes[1].set_ylabel("Mean Silhouette Score", fontsize=12)
axes[1].set_title("Silhouette Score vs K", fontsize=14)
axes[1].axvline(x=3, color="#e74c3c", linestyle="--", alpha=0.7, label="K = 3")
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Silhouette scores by K:")
for k, s in zip(sil_ks, sil_values):
    print(f"  K={k:2d}  ->  {s:.4f}")
```

Both plots agree: **K = 3** is the best choice for this dataset — inertia has a clear elbow and the silhouette score peaks at K = 3.

---

## 6. Scikit-learn KMeans

In practice you'll use scikit-learn's battle-tested implementation. It uses smarter **k-means++** initialization and runs multiple restarts (`n_init`) to avoid poor local minima. Let's compare with our scratch version:

```python
from sklearn.cluster import KMeans

km_sklearn = KMeans(n_clusters=3, random_state=42, n_init=10)
km_sklearn.fit(X)

print("=== Scikit-learn KMeans ===")
print(f"Inertia:          {km_sklearn.inertia_:.2f}")
print(f"Silhouette score: {silhouette_score(X, km_sklearn.labels_):.4f}")
print(f"Centroids:\n{km_sklearn.cluster_centers_}")
print()

print("=== Our scratch KMeans ===")
print(f"Inertia:          {km_scratch.inertia_:.2f}")
print(f"Silhouette score: {silhouette_score(X, km_scratch.labels_):.4f}")
print(f"Centroids:\n{km_scratch.centroids}")
```

The cluster labels may differ in numbering (label 0 in one could be label 2 in the other), but the **groupings themselves** should be nearly identical.

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors_map = np.array(["#e74c3c", "#2ecc71", "#3498db"])

axes[0].scatter(X[:, 0], X[:, 1], c=colors_map[km_scratch.labels_],
                edgecolors="k", s=50, alpha=0.7)
axes[0].scatter(km_scratch.centroids[:, 0], km_scratch.centroids[:, 1],
                c="gold", marker="X", s=250, edgecolors="k", linewidths=1.5, zorder=5)
axes[0].set_title("Our Scratch Implementation", fontsize=14)
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

axes[1].scatter(X[:, 0], X[:, 1], c=colors_map[km_sklearn.labels_],
                edgecolors="k", s=50, alpha=0.7)
axes[1].scatter(km_sklearn.cluster_centers_[:, 0], km_sklearn.cluster_centers_[:, 1],
                c="gold", marker="X", s=250, edgecolors="k", linewidths=1.5, zorder=5)
axes[1].set_title("Scikit-learn KMeans", fontsize=14)
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")

plt.suptitle("Scratch vs Scikit-learn — Side by Side", fontsize=15, y=1.01)
plt.tight_layout()
plt.show()
```

---

## 7. Practical Tips

### When K-Means Works Well

K-Means works best when clusters are:
- **Spherical (isotropic):** roughly the same spread in every direction
- **Similar in size:** very uneven cluster sizes can pull centroids away from smaller groups
- **Well-separated:** heavily overlapping clusters confuse the algorithm

### Feature Scaling

K-Means relies on Euclidean distance. If one feature has a range of 0–1 and another 0–10,000, the second feature will dominate. **Always standardize your features** (e.g., `StandardScaler`) before clustering.

### Multiple Initializations

Scikit-learn's `n_init` parameter (default 10) runs K-Means multiple times with different random seeds and keeps the result with the lowest inertia. This greatly reduces the risk of a poor local minimum.

### When K-Means Fails

K-Means struggles with:
- **Non-convex shapes** (e.g., crescent moons, concentric rings) — consider DBSCAN or spectral clustering instead
- **Clusters with very different densities** — HDBSCAN handles this better
- **High-dimensional data** — distances become less meaningful (curse of dimensionality); apply dimensionality reduction first

---

## Summary

### Key Takeaways

1. **Unsupervised learning** discovers structure without labels. Clustering is its flagship task.
2. **K-Means** iterates between *assigning* points to the nearest centroid and *updating* centroids as cluster means until convergence.
3. **Inertia** measures within-cluster compactness; **silhouette score** balances compactness and separation.
4. The **elbow method** plots inertia vs K to find a natural number of clusters.
5. **Scikit-learn's KMeans** adds smart initialization (k-means++) and multiple restarts for robust results.
6. Always **scale features** before clustering, and remember that K-Means assumes spherical, similarly-sized clusters.

### What's Next

In the following notebooks we will:
- Explore **hierarchical clustering** and dendrograms
- Learn **DBSCAN** for density-based clustering
- Apply **dimensionality reduction** (PCA, t-SNE) for visualization

---

*Generated by Berta AI | Created by Luigi Pascal Rondanini*
