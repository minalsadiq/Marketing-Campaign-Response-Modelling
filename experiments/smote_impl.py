"""
Minimal, from-scratch SMOTE implementation (Chawla et al. 2002).
Written by hand because imbalanced-learn isn't installable in this sandbox
(no network access) -- not because hand-rolling is preferred practice.
If you have network access, swap this for imblearn.over_sampling.SMOTE;
the algorithm and results should be equivalent.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote_oversample(X, y, minority_label=1, k=5, random_state=42):
    """Generate synthetic minority-class samples until classes are balanced 1:1."""
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    X_min = X[y == minority_label]
    n_min, n_maj = (y == minority_label).sum(), (y != minority_label).sum()
    n_to_generate = n_maj - n_min
    if n_to_generate <= 0:
        return X, y

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X_min))).fit(X_min)
    _, neighbor_idx = nn.kneighbors(X_min)

    synthetic = np.zeros((n_to_generate, X.shape[1]))
    for i in range(n_to_generate):
        base_idx = rng.randint(0, len(X_min))
        # neighbor_idx[base_idx, 0] is the point itself; pick from the rest
        neigh_choices = neighbor_idx[base_idx, 1:]
        neigh_idx = neigh_choices[rng.randint(0, len(neigh_choices))]
        gap = rng.rand()
        synthetic[i] = X_min[base_idx] + gap * (X_min[neigh_idx] - X_min[base_idx])

    X_res = np.vstack([X, synthetic])
    y_res = np.concatenate([y, np.full(n_to_generate, minority_label)])
    return X_res, y_res
