import re
from nltk import wordpunct_tokenize

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize
import numpy as np

class HashtagTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True):
        self.normalize = norm

    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets):
        hashtag_counts = np.zeros((len(raw_tweets), 1))
        for i, tweet in enumerate(raw_tweets):
            hashtag = 0
            for token in wordpunct_tokenize(tweet):
                if re.match(r'#\w+', token):
                    hashtag += 1
            hashtag_counts[i] = hashtag
        vectorized = hashtag_counts
        return normalize(vectorized) if self.normalize else vectorized
