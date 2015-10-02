from nltk import wordpunct_tokenize

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize
import numpy as np

from utils import filters


class EmoticonTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True):
        self.norm = norm

    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets):
        vectorized = np.zeros((len(raw_tweets), 4))
        for i, tweet in enumerate(raw_tweets):
            happy, sad = 0, 0

            tokens = wordpunct_tokenize(tweet)
            for token in tokens:
                if filters.Happy_RE.match(token):
                    happy += 1
                if filters.Sad_RE.match(token):
                    sad += 1
            vectorized[i] = [happy, sad, filters.Happy_RE.match(tokens[-1]) is not None, filters.Sad_RE.match(tokens[-1]) is not None]
        return normalize(vectorized) if self.norm else vectorized

