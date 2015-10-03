from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

class POSTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True):
        self.vectorizer = None
        self.normalize = norm

    def fit(self, X, y=None, **fit_params):
        self.vectorizer = DictVectorizer().fit([TweeboCacher.tag_dict])
        return self

    def transform(self, raw_tweets, **transform_params):
        pos_counts = []
        for tweet in raw_tweets:
            try:
                tweet_tags_count = TweeboCacher.get_cached_pos_counts()[tweet]
            except KeyError:
                print('KeyError!')
                print(tweet)
                TweeboCacher.cache(raw_tweets)
                tweet_tags_count = TweeboCacher.get_cached_pos_counts()[tweet]
            pos_counts.append(tweet_tags_count)
        vectorized = self.vectorizer.transform(pos_counts)
        return normalize(vectorized) if self.normalize else vectorized
