from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize
from utils import filters


class HashtagTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True, preprocessor=None):
        self.normalize = norm
        self.preprocessor = preprocessor

    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets):
        hashtag_counts = []
        for i, tweet in enumerate(raw_tweets):
            if self.preprocessor:
                tweet = self.preprocessor(tweet)

            hashtag_counts.append([len(filters.hashtag_RE.findall(tweet)),
                                   len(filters.username_RE.findall(tweet)),
                                   len(filters.url_RE.findall(tweet))])
            #if sum(hashtag_counts[i]) != 0: print hashtag_counts[i], tweet
        return normalize(hashtag_counts) if self.normalize else hashtag_counts
