from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize
import numpy as np
from nltk.tokenize import wordpunct_tokenize


# from resources.tweebo_cacher import TweeboCacher


class AllcapsTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True):
        self.normalize = norm

    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets):
        allcaps_counts = np.zeros((len(raw_tweets), 1))
        for i, tweet in enumerate(raw_tweets):
            allcaps = 0
            for word in wordpunct_tokenize(tweet):
                if word.isupper():
                    allcaps += 1
            allcaps_counts[i] = allcaps
            # for token in TweeboCacher.get_cached_tokens()[tweet]:
            #     if token.isupper():
            #         allcaps += 1
            # allcaps_counts[i] = allcaps
        vectorized = allcaps_counts
        return normalize(vectorized) if self.normalize else vectorized
