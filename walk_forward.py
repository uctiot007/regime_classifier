import numpy as np
import matplotlib.pyplot as plt
from scaler import StandardScaler
from k_means import KMeans


def walk_forward_regimes(X, min_train_size=252, refit_every=21, k=3,
                          random_state=42, window_mode="expanding", lookback=252):
    """
    Point-in-time regime classification with NO lookahead bias.

    window_mode:
      "expanding" - each refit trains on ALL data seen so far (default).
                    Realistic for "we've been running this since inception
                    and never discard history," but rare regimes (like a
                    single crash) permanently destabilize their centroid,
                    since they never leave the training set and there's
                    rarely enough new similar data to re-anchor them.
      "rolling"   - each refit trains on only the last `lookback` days.
                    Old crash data eventually ages out, keeping the
                    amount of training data (and therefore each
                    centroid's stability) comparable at every refit.
                    Tradeoff: a crash that happened long ago is
                    "forgotten" by the model entirely once it exits
                    the window.

    Returns:
      labels: np.ndarray, len(X). Days before min_train_size are -1.
      centroid_history: list of dicts (fit_index, centroids_scaled,
                         centroids_original), one per refit.
    """
    n = len(X)
    labels = np.full(n, -1, dtype=int)
    centroid_history = []

    scaler = None
    model = None

    for i in range(min_train_size, n):
        if model is None or (i - min_train_size) % refit_every == 0:
            if window_mode == "rolling":
                train_X = X[max(0, i - lookback):i]
            else:
                train_X = X[:i]

            scaler = StandardScaler()
            train_X_scaled = scaler.fit_transform(train_X)

            model = KMeans(k=k, random_state=random_state)
            model.fit(train_X_scaled)

            centroid_history.append({
                "fit_index": i,
                "centroids_scaled": model.centroids.copy(),
                "centroids_original": scaler.inverse_transform(model.centroids),
            })

        x_today_scaled = scaler.transform(X[i:i + 1])
        labels[i] = model.predict(x_today_scaled)[0]

    return labels, centroid_history


def cluster_persistence(labels):
    """
    switch_rate: fraction of days where label differs from previous day.
    run_lengths: consecutive-day streaks within the same regime.
    Excludes unclassified (-1) days.
    """
    valid = labels[labels != -1]
    if len(valid) < 2:
        raise ValueError("Not enough classified days to compute persistence.")

    switches = np.sum(valid[1:] != valid[:-1])
    switch_rate = switches / (len(valid) - 1)

    run_lengths = []
    current_run = 1
    for i in range(1, len(valid)):
        if valid[i] == valid[i - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)

    return {
        "switch_rate": switch_rate,
        "avg_run_length": float(np.mean(run_lengths)),
        "median_run_length": float(np.median(run_lengths)),
        "run_lengths": run_lengths,
    }


def centroid_drift(centroid_history):
    """
    Distance each centroid moves between consecutive refits, in
    original (unscaled) units. Centroids matched nearest-neighbor
    since KMeans cluster indices are arbitrary across refits.
    Returns array of shape (n_refits - 1, k).
    """
    drifts = []
    for prev, curr in zip(centroid_history[:-1], centroid_history[1:]):
        prev_c = prev["centroids_original"]
        curr_c = curr["centroids_original"]
        dists = np.linalg.norm(prev_c[:, np.newaxis, :] - curr_c[np.newaxis, :, :], axis=2)
        matched_idx = np.argmin(dists, axis=1)
        matched_dists = dists[np.arange(len(prev_c)), matched_idx]
        drifts.append(matched_dists)
    return np.array(drifts)


def plot_centroid_drift(centroid_hist, drift, dates, title="Max Centroid Drift per Refit", save_path=None):
    """Plots max centroid drift per refit over time; prints the largest spike's date."""
    max_drift_per_refit = drift.max(axis=1)
    refit_indices = [c["fit_index"] for c in centroid_hist[1:]]
    refit_dates = [dates[i] for i in refit_indices]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(refit_dates, max_drift_per_refit, marker="o", markersize=3, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Max drift across clusters")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()

    spike_idx = np.argmax(max_drift_per_refit)
    print(f"Largest drift spike: {max_drift_per_refit[spike_idx]:.4f} on {refit_dates[spike_idx]}")

    return refit_dates, max_drift_per_refit


def detection_lag(walk_forward_labels, dates, event_dates, min_run_length=3):
    """
    Measures how many trading days it took the walk-forward (no-lookahead)
    model to flag an event AFTER it actually started, i.e. the real-world
    delay before this system would have signaled "something changed."

    event_dates: known ground-truth event dates (e.g. the in-sample
                 model's crash-cluster dates, or externally known dates
                 like the actual COVID crash window).
    min_run_length: require the flagged regime to persist for at least
                 this many consecutive days, to avoid crediting a
                 1-day flicker as "detection."

    Returns the event's true start date, the date the walk-forward
    model first sustainably flagged a DIFFERENT regime at/after that
    point (proxy for "detected a regime change"), and the lag in
    trading days between them.
    """
    event_dates = sorted(event_dates)
    event_start = event_dates[0]

    date_to_idx = {d: i for i, d in enumerate(dates)}
    if event_start not in date_to_idx:
        raise ValueError("event_start not found in the provided dates index.")
    start_idx = date_to_idx[event_start]

    if start_idx == 0 or walk_forward_labels[start_idx - 1] == -1:
        baseline_label = None
    else:
        baseline_label = walk_forward_labels[start_idx - 1]

    detected_idx = None
    for i in range(start_idx, len(walk_forward_labels)):
        label = walk_forward_labels[i]
        if label == -1:
            continue
        if baseline_label is not None and label == baseline_label:
            continue
        # check it persists for min_run_length days (not a 1-day flicker)
        run = walk_forward_labels[i:i + min_run_length]
        if len(run) == min_run_length and np.all(run == label):
            detected_idx = i
            break

    if detected_idx is None:
        return {
            "event_start": event_start,
            "detected_date": None,
            "lag_trading_days": None,
            "note": "Walk-forward model never sustainably flagged a regime change for this event.",
        }

    detected_date = dates[detected_idx]
    lag_days = detected_idx - start_idx

    return {
        "event_start": event_start,
        "detected_date": detected_date,
        "lag_trading_days": lag_days,
    }