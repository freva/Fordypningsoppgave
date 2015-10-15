from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from storage import cache
import nltk


class POSTransformer(TransformerMixin, BaseEstimator):
    occurrences = cache.get("pos_tags", True)
    if not occurrences:
        raise Exception("PoS cache not found!")

    def __init__(self, norm=True, preprocessor=None):
        self.vectorizer = None
        self.normalize = norm
        self.preprocessor = preprocessor
        self.pos_dict = dict.fromkeys(POSTransformer.classes, 0)


    def fit(self, raw_tweets, y=None):
        self.vectorizer = DictVectorizer().fit([self.pos_dict])
        return self


    def transform(self, raw_tweets, **transform_params):
        updated = False
        pos_tabs = []

        for raw_tweet in raw_tweets:
            if raw_tweet in POSTransformer.occurrences:
                tag_frequencies = POSTransformer.occurrences[raw_tweet]
            else:
                raise Exception("Tweet \"" + raw_tweet + "\" not found in cache")
        vectorized = self.vectorizer.transform(pos_tabs)
        return normalize(vectorized) if self.normalize else vectorized
