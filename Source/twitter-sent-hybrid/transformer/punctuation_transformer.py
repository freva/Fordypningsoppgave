from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize


class PunctuationTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True, preprocessor=None):
        self.normalize = norm
        self.preprocessor = preprocessor

    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets):
        vectorized, search = [], ["!", "?", "\""]
        for i, tweet in enumerate(raw_tweets):
            if self.preprocessor:
                tweet = self.preprocessor(tweet)

            vectorized.append([float(tweet.count(i)) for i in search])
            #if sum(vectorized[-1]): print vectorized[-1], tweet
        return normalize(vectorized) if self.normalize else vectorized

