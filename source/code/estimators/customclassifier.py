from sklearn.base import BaseEstimator, ClassifierMixin


class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base):
        self.base = base

    def fit(self, X, y=None):
        self.base.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.base.predict(X)

    def predict_proba(self, X, y=None):
        return self.base.predict_proba(X)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

    def fit_predict_proba(self, X, y=None):
        self.fit(X, y)
        return self.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        return self.base.score(X, y, sample_weight)
