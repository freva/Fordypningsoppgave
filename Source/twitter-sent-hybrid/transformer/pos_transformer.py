from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from storage.tweebo_cache import TweeboCacher


class POSTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True, preprocessor=None):
        self.vectorizer = None
        self.normalize = norm
        self.preprocessor = preprocessor
        self.pos_dict = dict.fromkeys(['!', ',', '^', 'V', 'R', 'A', 'N', 'P', 'D', '$', 'E', '~', '@', 'O', 'T',
                                       'U', 'L', '&', 'X', '#', 'G', 'Z', 'Y', 'S', 'M'], 0)


    def fit(self, raw_tweets, y=None):
        self.vectorizer = DictVectorizer().fit([self.pos_dict])
        return self


    def transform(self, raw_tweets, **transform_params):
        pos_counts = []
        for tweet in raw_tweets:
            try:
                tweet_tags_count = TweeboCacher.get_cached_pos_counts()[tweet]
            except KeyError:
                for item in TweeboCacher.get_cached_pos_counts().keys():
                    if "More footage" in item:
                        print item
                print "Not found in cache:", tweet
                raise Exception('KeyError!')

            pos_counts.append(tweet_tags_count)
        vectorized = self.vectorizer.transform(pos_counts)
        return normalize(vectorized) if self.normalize else vectorized