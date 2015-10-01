from sklearn.feature_extraction.text import TransformerMixin
import numpy

class WordCounter(TransformerMixin):

    def transform(self, X, **transform_args):
        return numpy.array([[line.count(" ")] for line in X])

    def fit(self, X, Y, **fit_params):
        return self



