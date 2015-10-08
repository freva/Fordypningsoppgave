import re

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize


class ElongationTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True, preprocessor=None):
        self.normalize = norm
        self.preprocessor = preprocessor


    def fit(self, raw_tweets, y=None):
        return self


    def transform(self, raw_tweets):
        vectorized, repeat_RE = [], re.compile(r"([a-zA-Z])\1{2,}")
        for i, tweet in enumerate(raw_tweets):
            if self.preprocessor:
                tweet = self.preprocessor(tweet)

            vectorized.append([float(len(repeat_RE.findall(tweet)))])
            #if sum(vectorized[i]): print vectorized[i], tweet
        return normalize(vectorized) if self.normalize else vectorized
