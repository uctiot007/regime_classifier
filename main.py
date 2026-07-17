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