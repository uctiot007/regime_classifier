import numpy as np
import matplotlib.pyplot as plt


def plot_centroid_drift(centroid_hist, drift, dates, save_path=None):
    """
    Plots max centroid drift per refit over time, so you can visually
    check whether large drift events line up with known market stress
    periods (e.g. COVID crash) rather than occurring randomly.

    dates: the full date index used to build X (same one passed into
           walk_forward_regimes) — used to look up the actual calendar
           date of each refit.
    """
    max_drift_per_refit = drift.max(axis=1)  # shape (n_refits - 1,)

    # refit dates: skip the first refit (no prior refit to compare against)
    refit_indices = [c["fit_index"] for c in centroid_hist[1:]]
    refit_dates = [dates[i] for i in refit_indices]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(refit_dates, max_drift_per_refit, marker="o", markersize=3, linewidth=1)
    ax.set_title("Max Centroid Drift per Refit (original feature units)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Max drift across clusters")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()

    # Report the single largest drift event and its date
    spike_idx = np.argmax(max_drift_per_refit)
    print(f"Largest drift spike: {max_drift_per_refit[spike_idx]:.4f}")
    print(f"Occurred at refit date: {refit_dates[spike_idx]}")

    return refit_dates, max_drift_per_refit