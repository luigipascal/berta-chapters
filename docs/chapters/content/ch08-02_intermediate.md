# Ch 8: Unsupervised Learning - Intermediate

**Track**: Practitioner | [Try code in Playground](../../playground.md) | [Back to chapter overview](../chapter-08.md)


!!! tip "Read online or run locally"
    You can read this content here on the web. To run the code interactively,
    either use the [Playground](../../playground.md) or clone the repo and open
    `chapters/chapter-08-unsupervised-learning/notebooks/02_intermediate.ipynb` in Jupyter.

---

# Chapter 8: Unsupervised Learning
## Notebook 02 - Intermediate: Advanced Clustering

Beyond K-Means: hierarchical clustering, density-based methods, and Gaussian mixtures for real-world data shapes.

**What you'll learn:**
- Hierarchical (agglomerative) clustering and dendrograms
- DBSCAN for density-based clustering
- Gaussian Mixture Models (GMMs)
- Comparing clustering algorithms on different data shapes

**Time estimate:** 2.5 hours

**Try it yourself:** Experiment with different linkage methods (single, complete, average, ward) on the hierarchical clustering example. Change `eps` and `min_samples` in DBSCAN to see how they affect cluster formation.

**Common mistakes:** Using K-Means on non-convex shapes (e.g., moons), ignoring the k-distance graph when tuning DBSCAN, or assuming spherical clusters when data is elliptical.

---

## 1. Hierarchical Clustering

Hierarchical clustering builds a tree of clusters instead of requiring a fixed number of clusters up front. The **agglomerative (bottom-up)** approach proceeds as follows:

1. **Start** — treat every data point as its own single-point cluster.
2. **Merge** — find the two closest clusters and merge them into one.
3. **Repeat** — keep merging until only a single cluster remains (or until a stopping criterion is met).

The result is a hierarchy that can be visualised as a **dendrogram** — a tree diagram showing the order and distance of each merge.

### Linkage criteria

"Distance between two clusters" can be measured in several ways:

| Linkage | Definition | Tendency |
|---------|-----------|----------|
| **Single** | Minimum distance between any pair of points across two clusters | Produces elongated, chain-like clusters |
| **Complete** | Maximum distance between any pair of points across two clusters | Produces compact, roughly equal-sized clusters |
| **Average** | Mean distance between all pairs of points across two clusters | Compromise between single and complete |
| **Ward** | Minimises the total within-cluster variance at each merge | Tends to produce equally sized, spherical clusters |

Ward linkage is the most commonly used default and works well when clusters are roughly spherical.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

np.random.seed(42)

# Generate synthetic data with 4 well-separated clusters
X_hier, y_hier = make_blobs(
    n_samples=200, centers=4, cluster_std=0.8, random_state=42
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel — raw data
axes[0].scatter(X_hier[:, 0], X_hier[:, 1], s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
axes[0].set_title('Raw Data (200 points, 4 clusters)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# Right panel — dendrogram using Ward linkage
Z_ward = linkage(X_hier, method='ward')
dendrogram(
    Z_ward,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90,
    leaf_font_size=8,
    ax=axes[1],
    color_threshold=12
)
axes[1].set_title('Dendrogram (Ward Linkage, truncated to 30 leaves)')
axes[1].set_xlabel('Cluster (size)')
axes[1].set_ylabel('Merge Distance')
axes[1].axhline(y=12, color='r', linestyle='--', label='Cut at distance = 12')
axes[1].legend()

plt.tight_layout()
plt.show()
```

The dendrogram shows the full merge history. By drawing a horizontal cut line we decide how many clusters to keep — each vertical line that crosses the cut corresponds to one cluster.

### Comparing linkage methods

Let's visualise how the four linkage types partition the same dataset.

```python
linkage_methods = ['single', 'complete', 'average', 'ward']
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

for ax, method in zip(axes, linkage_methods):
    Z = linkage(X_hier, method=method)
    labels = fcluster(Z, t=4, criterion='maxclust')
    scatter = ax.scatter(
        X_hier[:, 0], X_hier[:, 1],
        c=labels, cmap='viridis', s=30, alpha=0.7, edgecolors='k', linewidths=0.3
    )
    ax.set_title(f'{method.capitalize()} linkage')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.suptitle('Agglomerative Clustering — 4 Linkage Methods (k=4)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

```python
# Scikit-learn's AgglomerativeClustering with Ward linkage
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
agg_labels = agg.fit_predict(X_hier)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(
    X_hier[:, 0], X_hier[:, 1],
    c=y_hier, cmap='tab10', s=40, alpha=0.7, edgecolors='k', linewidths=0.3
)
axes[0].set_title('Ground-Truth Labels')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

axes[1].scatter(
    X_hier[:, 0], X_hier[:, 1],
    c=agg_labels, cmap='tab10', s=40, alpha=0.7, edgecolors='k', linewidths=0.3
)
axes[1].set_title('AgglomerativeClustering (Ward, k=4)')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

print(f"Cluster sizes: {np.bincount(agg_labels)}")
```

---

## 2. DBSCAN

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) takes a fundamentally different approach to clustering:

- It does **not** require the number of clusters in advance.
- It defines clusters as **dense regions** separated by sparse regions.
- Points that don't belong to any dense region are labelled as **noise** (label = -1).

### Key parameters

| Parameter | Meaning |
|-----------|--------|
| `eps` (\( \varepsilon \)) | Maximum distance between two points for them to be considered neighbours |
| `min_samples` | Minimum number of points within \( \varepsilon \)-distance to form a dense region |

### Point types

- **Core point** — has at least `min_samples` neighbours within \( \varepsilon \).
- **Border point** — within \( \varepsilon \) of a core point but doesn't have enough neighbours itself.
- **Noise point** — neither core nor border; isolated outliers.

DBSCAN can discover clusters of **arbitrary shape** and naturally identifies outliers — something centroid-based methods like K-Means cannot do.

```python
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

np.random.seed(42)

# Generate two moons (non-convex dataset)
X_moons, y_moons = make_moons(n_samples=500, noise=0.08, random_state=42)

# Apply DBSCAN and K-Means
db_moons = DBSCAN(eps=0.2, min_samples=5).fit(X_moons)
km_moons = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_moons)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='coolwarm', s=20, alpha=0.7)
axes[0].set_title('Ground Truth')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=km_moons.labels_, cmap='coolwarm', s=20, alpha=0.7)
axes[1].scatter(km_moons.cluster_centers_[:, 0], km_moons.cluster_centers_[:, 1],
                marker='X', s=200, c='black', edgecolors='white', linewidths=1.5)
axes[1].set_title('K-Means (k=2) — Fails on non-convex shapes')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

axes[2].scatter(X_moons[:, 0], X_moons[:, 1], c=db_moons.labels_, cmap='coolwarm', s=20, alpha=0.7)
axes[2].set_title('DBSCAN (eps=0.2) — Correctly separates crescents')
axes[2].set_xlabel('Feature 1')
axes[2].set_ylabel('Feature 2')

plt.suptitle('K-Means vs DBSCAN on the Moons Dataset', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

---

## 3. Choosing DBSCAN Parameters

Picking `eps` and `min_samples` can be tricky. A practical heuristic:

1. Set `min_samples` \( \approx 2 \times \) number of features (a reasonable default).
2. For each point compute the distance to its **k-th nearest neighbour** (k = `min_samples`).
3. Sort these distances and plot them — the **k-distance graph**.
4. Look for the "elbow" — the point where the curve bends sharply upward. The distance at that elbow is a good candidate for `eps`.

```python
from sklearn.neighbors import NearestNeighbors

k = 5  # same as min_samples
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_moons)
distances, _ = nn.kneighbors(X_moons)

k_distances = np.sort(distances[:, k - 1])[::-1]

plt.figure(figsize=(10, 5))
plt.plot(k_distances, linewidth=1.5)
plt.axhline(y=0.2, color='r', linestyle='--', label='eps = 0.2 (our choice)')
plt.title(f'k-Distance Graph (k={k}) — Elbow Indicates Good eps')
plt.xlabel('Points (sorted by descending k-distance)')
plt.ylabel(f'Distance to {k}-th Nearest Neighbour')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

```python
# Effect of different eps values on DBSCAN results
eps_values = [0.05, 0.1, 0.2, 0.3, 0.5]
fig, axes = plt.subplots(1, len(eps_values), figsize=(22, 4))

for ax, eps in zip(axes, eps_values):
    db = DBSCAN(eps=eps, min_samples=5).fit(X_moons)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k_label, col in zip(sorted(unique_labels), colors):
        if k_label == -1:
            col = [0, 0, 0, 1]  # black for noise
        mask = labels == k_label
        ax.scatter(X_moons[mask, 0], X_moons[mask, 1], c=[col], s=15, alpha=0.7)

    ax.set_title(f'eps={eps}\n{n_clusters} clusters, {n_noise} noise')
    ax.set_xlabel('Feature 1')

axes[0].set_ylabel('Feature 2')
plt.suptitle('Effect of eps on DBSCAN (min_samples=5)', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()
```

**Observations:**
- **eps too small** (0.05) → most points classified as noise; many tiny clusters.
- **eps just right** (0.2) → two clean crescent clusters with very little noise.
- **eps too large** (0.5) → everything merges into a single cluster.

The k-distance graph helps you find that sweet spot without trial and error.

---

## 4. Gaussian Mixture Models

A **Gaussian Mixture Model** assumes that the data is generated from a mixture of a finite number of Gaussian (normal) distributions with unknown parameters.

### GMM vs K-Means

| Aspect | K-Means | GMM |
|--------|---------|-----|
| Cluster assignment | **Hard** — each point belongs to exactly one cluster | **Soft** — each point has a probability for every cluster |
| Cluster shape | Spherical (Voronoi cells) | Elliptical (full covariance matrices) |
| Outlier handling | None — every point is assigned | Naturally down-weights low-probability points |
| Output | Cluster label | Probability vector over all clusters |

GMMs are fit using the **Expectation-Maximisation (EM)** algorithm:
1. **E-step** — compute the probability that each point belongs to each Gaussian component.
2. **M-step** — update each component's mean, covariance, and weight to maximise log-likelihood.
3. Repeat until convergence.

```python
from sklearn.mixture import GaussianMixture

np.random.seed(42)

# Create elongated / elliptical clusters that challenge K-Means
n_per_cluster = 200
cov1 = [[2.0, 1.5], [1.5, 1.5]]
cov2 = [[1.5, -1.2], [-1.2, 1.5]]
cov3 = [[0.5, 0.0], [0.0, 2.5]]

cluster1 = np.random.multivariate_normal([0, 0], cov1, n_per_cluster)
cluster2 = np.random.multivariate_normal([5, 5], cov2, n_per_cluster)
cluster3 = np.random.multivariate_normal([8, 0], cov3, n_per_cluster)

X_gmm = np.vstack([cluster1, cluster2, cluster3])
y_gmm_true = np.array([0]*n_per_cluster + [1]*n_per_cluster + [2]*n_per_cluster)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Ground truth
axes[0].scatter(X_gmm[:, 0], X_gmm[:, 1], c=y_gmm_true, cmap='tab10', s=15, alpha=0.6)
axes[0].set_title('Ground Truth (Elliptical Clusters)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# K-Means
km_gmm = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_gmm)
axes[1].scatter(X_gmm[:, 0], X_gmm[:, 1], c=km_gmm.labels_, cmap='tab10', s=15, alpha=0.6)
axes[1].scatter(km_gmm.cluster_centers_[:, 0], km_gmm.cluster_centers_[:, 1],
                marker='X', s=200, c='black', edgecolors='white', linewidths=1.5)
axes[1].set_title('K-Means (k=3) — Spherical assumption')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

# GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_gmm)
gmm_labels = gmm.predict(X_gmm)
axes[2].scatter(X_gmm[:, 0], X_gmm[:, 1], c=gmm_labels, cmap='tab10', s=15, alpha=0.6)
axes[2].set_title('GMM (3 components) — Elliptical fit')
axes[2].set_xlabel('Feature 1')
axes[2].set_ylabel('Feature 2')

plt.suptitle('K-Means vs GMM on Elliptical Clusters', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

```python
# Visualise GMM probability contours
x_min, x_max = X_gmm[:, 0].min() - 2, X_gmm[:, 0].max() + 2
y_min, y_max = X_gmm[:, 1].min() - 2, X_gmm[:, 1].max() + 2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

log_prob = gmm.score_samples(grid_points)
log_prob = log_prob.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 7))
ax.contourf(xx, yy, np.exp(log_prob), levels=30, cmap='YlOrRd', alpha=0.6)
ax.contour(xx, yy, np.exp(log_prob), levels=10, colors='darkred', linewidths=0.5, alpha=0.5)
ax.scatter(X_gmm[:, 0], X_gmm[:, 1], c=gmm_labels, cmap='tab10', s=10, alpha=0.7,
           edgecolors='k', linewidths=0.2)

for i in range(gmm.n_components):
    ax.scatter(gmm.means_[i, 0], gmm.means_[i, 1],
               marker='+', s=300, c='black', linewidths=3)

ax.set_title('GMM Probability Density Contours')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
plt.tight_layout()
plt.show()
```

### Model selection with BIC and AIC

How many Gaussian components should we use? We can use information criteria:

- **BIC** (Bayesian Information Criterion) — penalises model complexity more heavily.
- **AIC** (Akaike Information Criterion) — lighter penalty.

**Lower is better** for both. We fit GMMs with different numbers of components and pick the one with the lowest BIC (or AIC).

```python
n_components_range = range(1, 10)
bic_scores = []
aic_scores = []

for n in n_components_range:
    gmm_test = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm_test.fit(X_gmm)
    bic_scores.append(gmm_test.bic(X_gmm))
    aic_scores.append(gmm_test.aic(X_gmm))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(list(n_components_range), bic_scores, 'bo-', label='BIC', linewidth=2)
ax.plot(list(n_components_range), aic_scores, 'rs--', label='AIC', linewidth=2)
ax.axvline(x=3, color='green', linestyle=':', alpha=0.7, label='True number of components (3)')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Score (lower is better)')
ax.set_title('GMM Model Selection: BIC and AIC')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Best BIC at n_components = {np.argmin(bic_scores) + 1}")
print(f"Best AIC at n_components = {np.argmin(aic_scores) + 1}")
```

---

## 5. Algorithm Comparison

Let's put all four algorithms head-to-head on three different data geometries:

1. **Blobs** — well-separated spherical clusters
2. **Moons** — two interleaving crescents
3. **Varied-variance blobs** — spherical clusters with very different densities

```python
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

n_samples = 500

# Dataset 1: standard blobs
X_blobs, y_blobs = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=42)

# Dataset 2: moons
X_moons2, y_moons2 = make_moons(n_samples=n_samples, noise=0.07, random_state=42)

# Dataset 3: varied-variance blobs
X_varied, y_varied = make_blobs(
    n_samples=n_samples, centers=3, cluster_std=[0.5, 2.5, 1.0], random_state=42
)

datasets = [
    ('Blobs', X_blobs, {'n_clusters': 3, 'eps': 1.0}),
    ('Moons', X_moons2, {'n_clusters': 2, 'eps': 0.2}),
    ('Varied', X_varied, {'n_clusters': 3, 'eps': 1.5}),
]

fig, axes = plt.subplots(3, 4, figsize=(22, 15))

for row, (name, X, params) in enumerate(datasets):
    X_scaled = StandardScaler().fit_transform(X)
    n_c = params['n_clusters']
    eps = params['eps']

    # K-Means
    km = KMeans(n_clusters=n_c, random_state=42, n_init=10).fit(X_scaled)
    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=n_c, linkage='ward').fit(X_scaled)
    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=5).fit(X_scaled)
    # GMM
    gm = GaussianMixture(n_components=n_c, random_state=42).fit(X_scaled)

    results = [
        ('K-Means', km.labels_),
        ('Agglomerative', agg.labels_),
        ('DBSCAN', db.labels_),
        ('GMM', gm.predict(X_scaled)),
    ]

    for col, (algo_name, labels) in enumerate(results):
        ax = axes[row, col]
        unique_labels = set(labels)
        n_clust = len(unique_labels) - (1 if -1 in unique_labels else 0)

        noise_mask = labels == -1
        ax.scatter(X_scaled[~noise_mask, 0], X_scaled[~noise_mask, 1],
                   c=labels[~noise_mask], cmap='viridis', s=12, alpha=0.7)
        if noise_mask.any():
            ax.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1],
                       c='red', marker='x', s=15, alpha=0.5, label='noise')
            ax.legend(fontsize=8)

        if row == 0:
            ax.set_title(algo_name, fontsize=13, fontweight='bold')
        ax.set_ylabel(f'{name}' if col == 0 else '', fontsize=12)
        ax.text(0.02, 0.98, f'{n_clust} cluster(s)',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.suptitle('Algorithm Comparison Across Data Geometries', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
```

---

## Summary

### When to use each algorithm

| Algorithm | Best for | Weaknesses | Must specify k? |
|-----------|----------|------------|-----------------|
| **K-Means** | Large datasets with spherical clusters | Cannot handle non-convex shapes; sensitive to outliers | Yes |
| **Agglomerative Clustering** | Small-to-medium datasets; exploring hierarchy | \( O(n^3) \) time complexity; hard to scale | Yes (or cut dendrogram) |
| **DBSCAN** | Arbitrary shapes; datasets with noise/outliers | Sensitive to `eps`; struggles with varying densities | No |
| **Gaussian Mixture Model** | Elliptical clusters; need soft assignments | Assumes Gaussian components; sensitive to initialisation | Yes |

### Rules of thumb

1. **Start simple:** try K-Means first. If results look poor, consider the data geometry.
2. **Non-convex shapes?** → Use DBSCAN.
3. **Elliptical or overlapping clusters?** → Use GMM.
4. **Need a hierarchy or dendrogram?** → Use Agglomerative Clustering.
5. **Noisy data with outliers?** → DBSCAN naturally handles noise.
6. **Need probability estimates?** → GMM provides soft assignments.

---
*Generated by Berta AI | Created by Luigi Pascal Rondanini*
