from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder


class CustomLabelEncoder(TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()

    def transform(self, X, y=None, **kwargs):
        return self.encoder.transform(y)

    def fit(self, X, y=None, **kwargs):
        self.encoder.fit(y)

    def fit_transform(self, X, y=None, **kwargs):
        self.encoder.fit(y)
        return self.encoder.transform(y)
