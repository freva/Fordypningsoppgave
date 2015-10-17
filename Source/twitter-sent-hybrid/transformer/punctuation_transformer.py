from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize
import re


class PunctuationTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True, preprocessors=None):
        self.normalize = norm
        self.preprocessors = preprocessors

    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets):
        vectorized, search = [], ["!", "?", "."]
        repeat_alpha_RE = re.compile(r"([a-zA-Z])\1{2,}")
        repeat_punkt_re = re.compile(r"[!?.,]{2,}")

        for tweet in raw_tweets:
            for preprocessor in self.preprocessors:
                tweet = preprocessor(tweet)

            vectorized.append([float(tweet.count(i)) for i in search] +
                              [len(repeat_alpha_RE.findall(tweet)), len(repeat_punkt_re.findall(tweet))])
            #if sum(vectorized[-1]): print vectorized[-1], tweet
        return normalize(vectorized) if self.normalize else vectorized

