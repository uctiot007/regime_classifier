from data_loader import fetch_data
from features import engineer_features
from scaler import StandardScaler
from k_means import KMeans

df = fetch_data(ticker="SPY", start="2015-01-01", end="2025-01-01")
df = engineer_features(df)

X = df[["vol_5d", "vol_21d", "ma_return_10d"]].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(k=3, random_state=42)
km.fit(X_scaled)

df["regime"] = km.labels

print(df["regime"].value_counts())
print(df.groupby("regime")[["vol_5d", "vol_21d", "ma_return_10d"]].mean())

from visualizer import plot_price_by_regime, plot_regime_background

plot_price_by_regime(df, save_path="regime_scatter.png")
plot_regime_background(df, save_path="regime_bands.png")

from analysis import elbow_analysis, cluster_profile

elbow_analysis(X_scaled, save_path="elbow.png")
cluster_profile(df)

from walk_forward import walk_forward_regimes, cluster_persistence, centroid_drift, plot_centroid_drift, detection_lag

# Expanding 
wf_labels_exp, hist_exp = walk_forward_regimes(X, min_train_size=252, refit_every=5, k=3, window_mode="expanding")
drift_exp = centroid_drift(hist_exp)
plot_centroid_drift(hist_exp, drift_exp, df.index, title="Centroid Drift — Expanding Window")

# Rolling 
wf_labels_roll, hist_roll = walk_forward_regimes(X, min_train_size=252, refit_every=5, k=3, window_mode="rolling", lookback=252)
drift_roll = centroid_drift(hist_roll)
plot_centroid_drift(hist_roll, drift_roll, df.index, title="Centroid Drift — Rolling Window (1yr lookback)")

# Detection lag — how many trading days after the real crash started did the model catch it?
crash_dates = df[df["regime"] == 2].index.tolist()  # from your original in-sample model
lag_result = detection_lag(wf_labels_exp, df.index, crash_dates, min_run_length=3)
print("Detection lag (expanding):", lag_result)

lag_result_roll = detection_lag(wf_labels_roll, df.index, crash_dates, min_run_length=3)
print("Detection lag (rolling):", lag_result_roll)

import pandas as pd

target_date = pd.Timestamp("2020-02-19")
snapped_date = df.index[df.index >= target_date][0]
print("Using date:", snapped_date)

lag_result_exp = detection_lag(wf_labels_exp, df.index, [snapped_date], min_run_length=3)
lag_result_roll = detection_lag(wf_labels_roll, df.index, [snapped_date], min_run_length=3)

print("Detection lag vs real market top (expanding):", lag_result_exp)
print("Detection lag vs real market top (rolling):", lag_result_roll)