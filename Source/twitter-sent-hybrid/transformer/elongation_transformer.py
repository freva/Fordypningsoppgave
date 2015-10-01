import re

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize
import numpy as np
from nltk.tokenize import wordpunct_tokenize

class ElongationTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True):
        self.normalize = norm

    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets):
        elong_counts = np.zeros((len(raw_tweets), 1))
        for i, tweet in enumerate(raw_tweets):
            elong = 0
            for word in wordpunct_tokenize(tweet):
                if re.search(r'([a-zA-Z])\1{3,}', word):
                    elong += 1
            elong_counts[i] = elong
        vectorized = elong_counts
        return normalize(vectorized) if self.normalize else vectorized
