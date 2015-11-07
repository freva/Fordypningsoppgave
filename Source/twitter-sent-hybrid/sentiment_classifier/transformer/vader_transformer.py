from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment

class VaderTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True, preprocessors=None):
        self.normalize = norm
        self.preprocessors = preprocessors

    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets):
        out = []

        for tweet in raw_tweets:
            for preprocessor in self.preprocessors:
                tweet = preprocessor(tweet)

            out.append(vaderSentiment(tweet.encode("ascii", "ignore")).values())
            #if sum(vectorized[-1]): print vectorized[-1], tweet
        return normalize(out) if self.normalize else out