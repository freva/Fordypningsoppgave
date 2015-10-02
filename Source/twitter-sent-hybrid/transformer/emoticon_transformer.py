from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize
import numpy as np

from utils import filters


class EmoticonTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True, preprocessor=None):
        self.norm = norm
        self.preprocessor = preprocessor

    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets):
        vectorized = np.zeros((len(raw_tweets), 2))
        for i, tweet in enumerate(raw_tweets):
            if self.preprocessor:
                tweet = self.preprocessor(tweet)

            happy = len(filters.Positive_RE.findall(tweet))
            sad = len(filters.Negative_RE.findall(tweet))

            vectorized[i] = [happy, sad]
            #if sum(vectorized[i]) != 0: print vectorized[i], tweet
        return normalize(vectorized) if self.norm else vectorized

