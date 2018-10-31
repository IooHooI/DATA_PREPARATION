from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelBinarizer


class CustomLabelBinarizer(TransformerMixin):
    def __init__(self, neg_label=0, pos_label=1, sparse_output=False):
        self.binarizer = LabelBinarizer(neg_label, pos_label, sparse_output)

    def transform(self, X, y=None, **kwargs):
        return self.binarizer.transform(y).ravel()

    def fit(self, X, y=None, **kwargs):
        self.binarizer.fit(y)

    def fit_transform(self, X, y=None, **kwargs):
        self.binarizer.fit(y)
        return self.binarizer.transform(y).ravel()
