from sklearn.feature_extraction.text import TransformerMixin
import numpy

class WordCounter(TransformerMixin):

    def transform(self, X, **transform_args):
        wordCounts = numpy.array([line.count(" ") for line in X])
        print wordCounts.shape
        return wordCounts

    def fit(self, X, Y, **fit_params):
        return self



