import numpy as np
from typing import Optional


class KMeans:
    """
    K-Means clustering, implemented from scratch.

    Expects scaled input (e.g. via a custom StandardScaler) — tol is
    chosen assuming features have roughly unit variance.
    """

    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        # 1. Init: pick k distinct rows from X as starting centroids
        random_idx = rng.choice(n_samples, self.k, replace=False)
        centroids = X[random_idx].copy()
        labels = None

        for _ in range(self.max_iters):
            # 2. Assign step: squared distances, shape (n_samples, k)
            #    no sqrt needed — doesn't change which centroid is closest
            diff = X[:, np.newaxis, :] - centroids        # (n_samples, k, n_features)
            sq_dist = np.sum(diff ** 2, axis=2)            # (n_samples, k)
            labels = np.argmin(sq_dist, axis=1)            # (n_samples,)

            # 3. Update step: recompute centroid = mean of its assigned points
            #    guard against empty clusters (keep old centroid if no points assigned)
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                for i in range(self.k)
            ])

            # 4. Convergence check: how far did centroids move this round?
            shift = np.linalg.norm(new_centroids - centroids, axis=1)
            centroids = new_centroids
            if np.max(shift) < self.tol:
                break

        self.centroids = centroids
        self.labels = labels
        return self

    def predict(self, X):
        assert self.centroids is not None, "Call fit() before predict()"
        diff = X[:, np.newaxis, :] - self.centroids
        sq_dist = np.sum(diff ** 2, axis=2)
        return np.argmin(sq_dist, axis=1)

    def inertia(self, X):
        """WCSS — total squared distance from each point to its assigned centroid.
        Useful for elbow-method comparison across different k."""
        assert self.centroids is not None and self.labels is not None, "Call fit() before inertia()"
        diff = X - self.centroids[self.labels]
        return np.sum(diff ** 2)