from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize
from utils import filters


class EmoticonTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True, preprocessors=None):
        self.norm = norm
        self.preprocessors = preprocessors

    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets):
        vectorized = []
        for tweet in raw_tweets:
            for preprocessor in self.preprocessors:
                tweet = preprocessor(tweet)

            happy = float(len(filters.Positive_RE.findall(tweet)))
            sad = float(len(filters.Negative_RE.findall(tweet)))

            vectorized.append([happy, sad])
            #if sum(vectorized[i]): print vectorized[i], tweet
        return normalize(vectorized) if self.norm else vectorized

