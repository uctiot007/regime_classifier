import numpy as np
import matplotlib.pyplot as plt

from k_means import KMeans


def elbow_analysis(X, k_range=range(1, 7), random_state=42, save_path=None):
    """
    Fits KMeans for each k in k_range, records WCSS (inertia).
    Returns a dict {k: inertia} and plots the elbow curve.
    """
    inertias = {}

    for k in k_range:
        km = KMeans(k=k, random_state=random_state)
        km.fit(X)
        inertias[k] = km.inertia(X)
        print(f"k={k}: inertia={inertias[k]:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(inertias.keys()), list(inertias.values()), marker="o")
    ax.set_title("Elbow Method: Inertia (WCSS) vs k")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_xticks(list(k_range))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved elbow plot to {save_path}")

    plt.show()
    return inertias


def cluster_profile(df, regime_col="regime", feature_cols=None):
    """
    Prints per-regime summary stats: count and mean of each feature.
    Useful for labeling clusters after the fact (e.g. 'this is the crash regime').
    """
    if feature_cols is None:
        feature_cols = ["vol_5d", "vol_21d", "ma_return_10d"]

    counts = df[regime_col].value_counts().sort_index()
    means = df.groupby(regime_col)[feature_cols].mean()

    print("Regime counts:")
    print(counts)
    print("\nRegime feature means:")
    print(means)

    return counts, means