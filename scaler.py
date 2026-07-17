import numpy as np


class StandardScaler:
    """
    Standardizes features to mean 0, std 1: z = (x - mean) / std
    Fit on training data, then use the same mean/std to transform
    any other data (avoids leakage if you later do train/test splits).
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)   # one mean per feature, shape (n_features,)
        self.std_ = X.std(axis=0)     # one std per feature, shape (n_features,)
        # guard against divide-by-zero if a feature is constant
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std_ + self.mean_