import pandas as pd


def engineer_features(df, price_col="Close"):
    """
    Computes regime-classification features from a price series:
      - daily return
      - 5-day realized volatility (rolling std of daily returns)
      - 21-day realized volatility (rolling std of daily returns)
      - 10-day moving average return

    Drops rows with NaN (from pct_change() and the rolling windows
    not having enough history yet at the start of the series).
    """
    df = df.copy()

    df["return"] = df[price_col].pct_change()
    df["vol_5d"] = df["return"].rolling(5).std()
    df["vol_21d"] = df["return"].rolling(21).std()
    df["ma_return_10d"] = df["return"].rolling(10).mean()

    df = df.dropna()
    return df