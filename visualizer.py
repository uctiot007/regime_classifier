import matplotlib.pyplot as plt
import matplotlib.dates as mdates


REGIME_COLORS = {
    0: "#2ca02c",   # green  -> typically the calm/steady regime
    1: "#ff7f0e",   # orange -> typically the choppy/correction regime
    2: "#d62728",   # red    -> typically the crash/high-vol regime
}

REGIME_LABELS = {
    0: "Regime 0",
    1: "Regime 1",
    2: "Regime 2",
}


def plot_price_by_regime(df, price_col="Close", regime_col="regime",
                          title="Price Colored by Market Regime", save_path=None):
    """
    Scatter-plots price over time, colored by cluster/regime label.
    Assumes df has a DatetimeIndex, a price column, and a regime column.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for regime in sorted(df[regime_col].unique()):
        subset = df[df[regime_col] == regime]
        ax.scatter(
            subset.index,
            subset[price_col],
            s=8,
            color=REGIME_COLORS.get(regime, "gray"),
            label=REGIME_LABELS.get(regime, f"Regime {regime}"),
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(price_col)
    ax.legend(markerscale=2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()
    return fig, ax


def plot_regime_background(df, price_col="Close", regime_col="regime",
                            title="Price with Regime Shading", save_path=None):
    """
    Alternative view: continuous price line with shaded background bands
    per regime. Reads more like a 'regime chart' than scattered points.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df[price_col], color="black", linewidth=1)

    # Shade contiguous regime runs as vertical spans
    regimes = df[regime_col].values
    dates = df.index
    start_idx = 0
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes[i] != regimes[start_idx]:
            ax.axvspan(
                dates[start_idx], dates[i - 1],
                color=REGIME_COLORS.get(regimes[start_idx], "gray"),
                alpha=0.25,
            )
            start_idx = i

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(price_col)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()
    return fig, ax